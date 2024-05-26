from __future__ import absolute_import
from __future__ import division
from tqdm import tqdm  #一个快速，可扩展的Python进度条
import json #一种轻量级的数据交换格式
import time
import os
import logging #logging是软件运行过程中跟踪一些时间发生的一种手段
import numpy as np
import tensorflow as tf
from code.model.agent import Agent
from code.options import read_options
from code.model.environment import env
import codecs #codecs专门用作编码转换
from collections import defaultdict
#Python中通过Key访问字典，当Key不存在时，会引发‘KeyError’异常。为了避免这种情况的发生，可以使用collections类中的defaultdict()方法来为字典提供默认值
import gc  #gc模块提供一个接口给开发者设置垃圾回收的选项
import resource  #resource 模块用于查询或修改当前系统资源限制设置
import sys   #提供对解释器使用或维护的一些变量的访问，以及与解释器强烈交互的函数
from code.model.baseline import ReactiveBaseline
from code.model.nell_eval import nell_eval
from scipy.misc import logsumexp as lse  ##logsumexp是先对矩阵以e次方求和取对数
import torch
from scipy.stats import pearsonr # pearsonr计算特征与目标变量之间的相关度




logger = logging.getLogger()
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class Trainer(object):
    def __init__(self, params):

        # transfer parameters to self
        for key, val in params.items(): setattr(self, key, val);

        self.agent = Agent(params)
        self.save_path = None
        self.train_environment = env(params, 'train')
        self.dev_test_environment = env(params, 'dev')
        self.test_test_environment = env(params, 'test')
        self.test_environment = self.dev_test_environment
        self.rev_relation_vocab = self.train_environment.grapher.rev_relation_vocab
        self.rev_entity_vocab = self.train_environment.grapher.rev_entity_vocab
        self.rev_tim_vocab = self.train_environment.grapher.rev_tim_vocab
        self.max_hits_at_10 = 0
        self.ePAD = self.entity_vocab['PAD']
        self.rPAD = self.relation_vocab['PAD']
        self.tPAD = self.tim_vocab['PAD']
        # optimize
        self.baseline = ReactiveBaseline(l=self.Lambda)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)


    def calc_reinforce_loss(self):
        loss = tf.stack(self.per_example_loss, axis=1)  # [B, T]  [2560,3]
        #q = loss.shape
        #with tf.Session() as sess:
            #print(sess.run(q))


        self.tf_baseline = self.baseline.get_baseline_value()
        # self.pp = tf.Print(self.tf_baseline)
        # multiply with rewards
        final_reward = self.cum_discounted_reward - self.tf_baseline
        #qq = final_reward.shape
        #with tf.Session() as sess:
            #print(sess.run(qq))
        # reward_std = tf.sqrt(tf.reduce_mean(tf.square(final_reward))) + 1e-5 # constant addded for numerical stability
        reward_mean, reward_var = tf.nn.moments(final_reward, axes=[0, 1])
        # Constant added for numerical stability
        reward_std = tf.sqrt(reward_var) + 1e-6
        final_reward = tf.div(final_reward - reward_mean, reward_std)


        loss = tf.multiply(loss, final_reward)  # [B, T]
        self.loss_before_reg = loss

        total_loss = tf.reduce_mean(loss) - self.decaying_beta * self.entropy_reg_loss(self.per_example_logits)  # scalar

        return total_loss

    def entropy_reg_loss(self, all_logits):
        all_logits = tf.stack(all_logits, axis=2)  # [B, MAX_NUM_ACTIONS, T]
        entropy_policy = - tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.exp(all_logits), all_logits), axis=1))  # scalar
        return entropy_policy

    def initialize(self, restore=None, sess=None):

        logger.info("Creating TF graph...")
        self.candidate_relation_sequence = []
        self.candidate_entity_sequence = []
        self.candidate_tim_sequence = []
        self.input_path = []
        self.input_path_tim = []
        self.first_state_of_test = tf.placeholder(tf.bool, name="is_first_state_of_test")
        self.query_relation = tf.placeholder(tf.int32, [None], name="query_relation")
        self.query_tim = tf.placeholder(tf.int32, [None], name="query_tim")
        self.range_arr = tf.placeholder(tf.int32, shape=[None, ])
        self.global_step = tf.Variable(0, trainable=False)
        self.decaying_beta = tf.train.exponential_decay(self.beta, self.global_step,
                                                   200, 0.90, staircase=False)
        self.entity_sequence = []

        # to feed in the discounted reward tensor
        self.cum_discounted_reward = tf.placeholder(tf.float32, [None, self.path_length],
                                                    name="cumulative_discounted_reward")



        for t in range(self.path_length):
            next_possible_relations = tf.placeholder(tf.int32, [None, self.max_num_actions],
                                                   name="next_relations_{}".format(t))
            next_possible_tims = tf.placeholder(tf.int32, [None, self.max_num_actions],
                                                     name="next_tims_{}".format(t))
            next_possible_entities = tf.placeholder(tf.int32, [None, self.max_num_actions],
                                                     name="next_entities_{}".format(t))
            input_label_relation = tf.placeholder(tf.int32, [None], name="input_label_relation_{}".format(t))
            input_label_tim = tf.placeholder(tf.int32, [None], name="input_label_tim_{}".format(t))
            start_entities = tf.placeholder(tf.int32, [None, ])
            self.input_path.append(input_label_relation)
            self.input_path_tim.append(input_label_tim)

            self.candidate_relation_sequence.append(next_possible_relations)
            self.candidate_tim_sequence.append(next_possible_tims)
            self.candidate_entity_sequence.append(next_possible_entities)
            self.entity_sequence.append(start_entities)
        self.loss_before_reg = tf.constant(0.0)
        self.per_example_loss, self.per_example_logits, self.action_idx, self.tim_idx = self.agent(
            self.candidate_relation_sequence, self.candidate_tim_sequence,
            self.candidate_entity_sequence, self.entity_sequence,
            self.input_path, self.input_path_tim,
            self.query_relation, self.query_tim, self.range_arr, self.first_state_of_test, self.path_length)


        self.loss_op = self.calc_reinforce_loss()

        # backprop
        self.train_op = self.bp(self.loss_op)

        # Building the test graph
        self.prev_state = tf.placeholder(tf.float32, self.agent.get_mem_shape(), name="memory_of_agent")
        self.prev_relation = tf.placeholder(tf.int32, [None, ], name="previous_relation")
        self.prev_tim = tf.placeholder(tf.int32, [None, ], name="previous_tim")
        self.query_embedding = tf.nn.embedding_lookup(self.agent.relation_lookup_table, self.query_relation)  # [B, 2D]
        self.query_embedding_tim = tf.nn.embedding_lookup(self.agent.tim_lookup_table, self.query_tim)
        layer_state = tf.unstack(self.prev_state, self.LSTM_layers)
        formated_state = [tf.unstack(s, 2) for s in layer_state]
        self.next_relations = tf.placeholder(tf.int32, shape=[None, self.max_num_actions])
        self.next_tims = tf.placeholder(tf.int32, shape=[None, self.max_num_actions])
        self.next_entities = tf.placeholder(tf.int32, shape=[None, self.max_num_actions])
        self.current_entities = tf.placeholder(tf.int32, shape=[None,])



        with tf.variable_scope("policy_steps_unroll") as scope:
            scope.reuse_variables()
            self.test_loss, test_state,self.test_logits, self.test_action_idx, self.test_tim_idx, self.chosen_relation, self.chosen_tim  = self.agent.step(
                self.next_relations, self.next_tims, self.next_entities, formated_state, self.prev_relation,
                self.prev_tim, self.query_embedding, self.query_embedding_tim,
                self.current_entities, self.input_path[0], self.input_path_tim[0],
                self.range_arr, self.first_state_of_test)

            self.test_state = tf.stack(test_state)

        logger.info('TF Graph creation done..')
        self.model_saver = tf.train.Saver(max_to_keep=2)

        # return the variable initializer Op.
        if not restore:
            return tf.global_variables_initializer()
        else:
            return  self.model_saver.restore(sess, restore)



    def initialize_pretrained_embeddings(self, sess):
        if self.pretrained_embeddings_action != '':
            embeddings = np.loadtxt(open(self.pretrained_embeddings_action))
            _ = sess.run((self.agent.relation_embedding_init),
                         feed_dict={self.agent.relation_embedding_placeholder: embeddings})
        if self.pretrained_embeddings_entity != '':
            embeddings = np.loadtxt(open(self.pretrained_embeddings_entity))
            _ = sess.run((self.agent.entity_embedding_init),
                         feed_dict={self.agent.entity_embedding_placeholder: embeddings})

        if self.pretrained_embeddings_tim != '':
            embeddings = np.loadtxt(open(self.pretrained_embeddings_tim))
            _ = sess.run((self.agent.tim_embedding_init),
                         feed_dict={self.agent.tim_embedding_placeholder: embeddings})

    def bp(self, cost):
        self.baseline.update(tf.reduce_mean(self.cum_discounted_reward))
        tvars = tf.trainable_variables()
        grads = tf.gradients(cost, tvars)
        grads, _ = tf.clip_by_global_norm(grads, self.grad_clip_norm)
        train_op = self.optimizer.apply_gradients(zip(grads, tvars))
        with tf.control_dependencies([train_op]):  # see https://github.com/tensorflow/tensorflow/issues/1899
            self.dummy = tf.constant(0)
        return train_op


    def calc_cum_discounted_reward(self, rewards):
        """
        calculates the cumulative discounted reward.
        :param rewards:
        :param T:
        :param gamma:
        :return:
        """
        running_add = np.zeros([rewards.shape[0]])  # [B]
        cum_disc_reward = np.zeros([rewards.shape[0], self.path_length])  # [B, T]
        cum_disc_reward[:,
        self.path_length - 1] = rewards  # set the last time step to the reward received at the last state
        for t in reversed(range(self.path_length)):
            running_add = self.gamma * running_add + cum_disc_reward[:, t]
            cum_disc_reward[:, t] = running_add
        return cum_disc_reward

    def gpu_io_setup(self):
        # create fetches for partial_run_setup
        fetches = self.per_example_loss  + self.action_idx +self.tim_idx + [self.loss_op] + self.per_example_logits + [self.dummy]
        feeds =  [self.first_state_of_test] + self.candidate_relation_sequence + self.candidate_tim_sequence + self.candidate_entity_sequence + self.input_path + self.input_path_tim + \
                [self.query_relation] + [self.query_tim] + [self.cum_discounted_reward] + [self.range_arr] + self.entity_sequence


        feed_dict = [{} for _ in range(self.path_length)]#_ 是占位符， 表示不在意变量的值 只是用于循环遍历n次

        feed_dict[0][self.first_state_of_test] = False
        feed_dict[0][self.query_relation] = None
        feed_dict[0][self.query_tim] = None
        feed_dict[0][self.range_arr] = np.arange(self.batch_size*self.num_rollouts)
        for i in range(self.path_length):
            feed_dict[i][self.input_path[i]] = np.zeros(self.batch_size * self.num_rollouts)  # placebo
            feed_dict[i][self.input_path_tim[i]] = np.zeros(self.batch_size * self.num_rollouts)  # placebo
            feed_dict[i][self.candidate_relation_sequence[i]] = None
            feed_dict[i][self.candidate_tim_sequence[i]] = None
            feed_dict[i][self.candidate_entity_sequence[i]] = None
            feed_dict[i][self.entity_sequence[i]] = None

        return fetches, feeds, feed_dict

    def train(self, sess):
        # import pdb
        # pdb.set_trace()
        fetches, feeds, feed_dict = self.gpu_io_setup()

        train_loss = 0.0
        start_time = time.time()
        self.batch_counter = 0
        for episode in self.train_environment.get_episodes():

            self.batch_counter += 1
            h = sess.partial_run_setup(fetches=fetches, feeds=feeds)
            feed_dict[0][self.query_relation] = episode.get_query_relation()
            feed_dict[0][self.query_tim] = episode.get_query_tim()

            # get initial state
            state = episode.get_state()


            # for each time step
            loss_before_regularization = []
            logits = []

            for i in range(self.path_length):
                feed_dict[i][self.candidate_relation_sequence[i]] = state['next_relations']
                feed_dict[i][self.candidate_tim_sequence[i]] = state['next_tims']
                feed_dict[i][self.candidate_entity_sequence[i]] = state['next_entities']
                feed_dict[i][self.entity_sequence[i]] = state['current_entities']
                per_example_loss, per_example_logits, action_idx,tim_idx = sess.partial_run(h, [self.per_example_loss[i], self.per_example_logits[i], self.action_idx[i],self.tim_idx[i]],feed_dict=feed_dict[i])
                loss_before_regularization.append(per_example_loss)
                logits.append(per_example_logits)
                # action = np.squeeze(action, axis=1)  # [B,]
                state = episode(action_idx)




            #for q in range(1, self.path_length):
                #aa = feed_dict[0][self.candidate_tim_sequence[0]]
                #aa = aa + feed_dict[i][self.candidate_tim_sequence[i]]


            loss_before_regularization = np.stack(loss_before_regularization, axis=1)

            # get the final reward from the environment
            rewards = episode.get_reward(action_idx)

            # computed cumulative discounted reward
            cum_discounted_reward = self.calc_cum_discounted_reward(rewards)  # [B, T]

            # backprop
            batch_total_loss, _ = sess.partial_run(h, [self.loss_op, self.dummy],
                                                   feed_dict={self.cum_discounted_reward: cum_discounted_reward})

            # print statistics
            train_loss = 0.95 * train_loss + 0.05 * batch_total_loss
            avg_reward = np.mean(rewards)
            # now reshape the reward to [orig_batch_size, num_rollouts], I want to calculate for how many of the
            # entity pair, atleast one of the path get to the right answer
            reward_reshape = np.reshape(rewards, (self.batch_size, self.num_rollouts))  # [orig_batch, num_rollouts]
            reward_reshape = np.sum(reward_reshape, axis=1)  # [orig_batch]
            reward_reshape = (reward_reshape > 0)
            num_ep_correct = np.sum(reward_reshape)
            if np.isnan(train_loss):
                raise ArithmeticError("Error in computing loss")

            logger.info("batch_counter: {0:4d}, num_hits: {1:7.4f}, avg. reward per batch {2:7.4f}, "
                        "num_ep_correct {3:4d}, avg_ep_correct {4:7.4f}, train loss {5:7.4f}".
                        format(self.batch_counter, np.sum(rewards), avg_reward, num_ep_correct,
                               (num_ep_correct / self.batch_size),
                               train_loss))

            if self.batch_counter%self.eval_every == 0:
                with open(self.output_dir + '/scores.txt', 'a') as score_file:
                    score_file.write("Score for iteration " + str(self.batch_counter) + "\n")
                os.mkdir(self.path_logger_file + "/" + str(self.batch_counter))
                self.path_logger_file_ = self.path_logger_file + "/" + str(self.batch_counter) + "/paths"



                self.test(sess, beam=True, print_paths=False)

            logger.info('Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            gc.collect()
            if self.batch_counter >= self.total_iterations:
                break

    def test(self, sess, beam=False, print_paths=False, save_model = True, auc = False):
        batch_counter = 0
        paths = defaultdict(list)



        answers = []

        feed_dict = {}
        all_final_reward_1 = 0
        all_final_reward_3 = 0
        all_final_reward_5 = 0
        all_final_reward_10 = 0
        all_final_reward_20 = 0
        auc = 0
        mrr = 0

        total_examples = self.test_environment.total_no_examples
        for episode in tqdm(self.test_environment.get_episodes()):
            batch_counter += 1

            temp_batch_size = episode.no_examples

            self.qr = episode.get_query_relation()
            self.qt = episode.get_query_tim()
            feed_dict[self.query_relation] = self.qr
            feed_dict[self.query_tim] = self.qt
            # set initial beam probs
            beam_probs = np.zeros((temp_batch_size * self.test_rollouts, 1))
            # get initial state
            state = episode.get_state()
            mem = self.agent.get_mem_shape()
            agent_mem = np.zeros((mem[0], mem[1], temp_batch_size*self.test_rollouts, mem[3]) ).astype('float32')
            previous_relation = np.ones((temp_batch_size * self.test_rollouts, ), dtype='int64') * self.relation_vocab[
                'DUMMY_START_RELATION']
            previous_tim = np.ones((temp_batch_size * self.test_rollouts, ), dtype='int64') * self.tim_vocab[
                'DUMMY_START_TIM']
            feed_dict[self.range_arr] = np.arange(temp_batch_size * self.test_rollouts)
            feed_dict[self.input_path[0]] = np.zeros(temp_batch_size * self.test_rollouts)
            feed_dict[self.input_path_tim[0]] = np.zeros(temp_batch_size * self.test_rollouts)

            ####logger code####
            if print_paths:
                self.entity_trajectory = []
                self.relation_trajectory = []
                self.tim_trajectory = []
            ####################

            self.log_probs = np.zeros((temp_batch_size*self.test_rollouts,)) * 1.0

            # for each time step
            for i in range(self.path_length):
                if i == 0:
                    feed_dict[self.first_state_of_test] = True
                feed_dict[self.next_relations] = state['next_relations']
                feed_dict[self.next_tims] = state['next_tims']
                feed_dict[self.next_entities] = state['next_entities']
                feed_dict[self.current_entities] = state['current_entities']
                feed_dict[self.prev_state] = agent_mem
                feed_dict[self.prev_relation] = previous_relation
                feed_dict[self.prev_tim] = previous_tim



                loss,agent_mem, test_scores, test_action_idx,test_tim_idx,  chosen_relation,chosen_tim = sess.run(
                        [self.test_loss, self.test_state, self.test_logits, self.test_action_idx,self.test_tim_idx,
                         self.chosen_relation,self.chosen_tim], feed_dict=feed_dict)



                if beam:
                    k = self.test_rollouts #100
                    new_scores = test_scores + beam_probs
                    if i == 0:
                        idx = np.argsort(new_scores)
                        idx = idx[:, -k:]
                        ranged_idx = np.tile([b for b in range(k)], temp_batch_size)
                        idx = idx[np.arange(k*temp_batch_size), ranged_idx]
                    else:
                        idx = self.top_k(new_scores, k)

                    y = idx//self.max_num_actions
                    x = idx%self.max_num_actions

                    y += np.repeat([b*k for b in range(temp_batch_size)], k)
                    state['current_entities'] = state['current_entities'][y]
                    state['next_relations'] = state['next_relations'][y,:]
                    state['next_tims'] = state['next_tims'][y, :]
                    state['next_entities'] = state['next_entities'][y, :]
                    agent_mem = agent_mem[:, :, y, :]
                    test_action_idx = x
                    test_tim_idx = x
                    chosen_relation = state['next_relations'][np.arange(temp_batch_size*k), x]
                    chosen_tim = state['next_tims'][np.arange(temp_batch_size * k), x]
                    beam_probs = new_scores[y, x]
                    beam_probs = beam_probs.reshape((-1, 1))
                    if print_paths:
                        for j in range(i):
                            self.entity_trajectory[j] = self.entity_trajectory[j][y]
                            self.relation_trajectory[j] = self.relation_trajectory[j][y]
                            self.tim_trajectory[j] = self.tim_trajectory[j][y]
                previous_relation = chosen_relation
                previous_tim = chosen_tim

                ####logger code####
                if print_paths:
                    self.entity_trajectory.append(state['current_entities'])
                    self.relation_trajectory.append(chosen_relation)
                    self.tim_trajectory.append(chosen_tim)
                ####################
                state = episode(test_action_idx)
                self.log_probs += test_scores[np.arange(self.log_probs.shape[0]), test_action_idx]

            if beam:
                self.log_probs = beam_probs

            ####Logger code####

            if print_paths:
                self.entity_trajectory.append(
                    state['current_entities'])




            # ask environment for final reward
            rewards = episode.get_reward(test_action_idx)  # [B*test_rollouts]

            with open(self.output_dir + '/scores_action.txt', 'a') as score_file:
                score_file.write("rewards: {0:7.4f}".format(rewards))
                score_file.write("\n")


            #rewards = np.resize(rewards,(3200, ))

            reward_reshape = np.reshape(rewards, (temp_batch_size, self.test_rollouts))  # [orig_batch, test_rollouts]
            self.log_probs = np.reshape(self.log_probs, (temp_batch_size, self.test_rollouts))
            sorted_indx = np.argsort(-self.log_probs)
            final_reward_1 = 0
            final_reward_3 = 0
            final_reward_5 = 0
            final_reward_10 = 0
            final_reward_20 = 0
            AP = 0
            ce = episode.state['current_entities'].reshape((temp_batch_size, self.test_rollouts))
            se = episode.start_entities.reshape((temp_batch_size, self.test_rollouts))
            for b in range(temp_batch_size):
                answer_pos = None
                seen = set()
                pos=0
                if self.pool == 'max':
                    for r in sorted_indx[b]:
                        if reward_reshape[b,r] == self.positive_reward:
                            answer_pos = pos
                            break
                        if ce[b, r] not in seen:
                            seen.add(ce[b, r])
                            pos += 1
                if self.pool == 'sum':
                    scores = defaultdict(list)
                    answer = ''
                    for r in sorted_indx[b]:
                        scores[ce[b,r]].append(self.log_probs[b,r])
                        if reward_reshape[b,r] == self.positive_reward:
                            answer = ce[b,r]
                    final_scores = defaultdict(float)
                    for e in scores:
                        final_scores[e] = lse(scores[e])
                    sorted_answers = sorted(final_scores, key=final_scores.get, reverse=True)
                    if answer in  sorted_answers:
                        answer_pos = sorted_answers.index(answer)
                    else:
                        answer_pos = None


                if answer_pos != None:
                    if answer_pos < 20:
                        final_reward_20 += 1
                        if answer_pos < 10:
                            final_reward_10 += 1
                            if answer_pos < 5:
                                final_reward_5 += 1
                                if answer_pos < 3:
                                    final_reward_3 += 1
                                    if answer_pos < 1:
                                        final_reward_1 += 1
                if answer_pos == None:
                    AP += 0
                else:
                    AP += 1.0/((answer_pos+1))
                if print_paths:
                    qr = self.train_environment.grapher.rev_relation_vocab[self.qr[b * self.test_rollouts]]
                    qt = self.train_environment.grapher.rev_tim_vocab[self.qt[b * self.test_rollouts]]
                    start_e = self.rev_entity_vocab[episode.start_entities[b * self.test_rollouts]]
                    end_e = self.rev_entity_vocab[episode.end_entities[b * self.test_rollouts]]
                    paths[str(qr)].append(str(start_e) + "\t" + str(end_e) + "\n")
                    paths[str(qr)].append("Reward:" + str(1 if answer_pos != None and answer_pos < 10 else 0) + "\n")
                    paths[str(qt)].append(str(start_e) + "\t" + str(end_e) + "\n")
                    paths[str(qt)].append("Reward:" + str(1 if answer_pos != None and answer_pos < 10 else 0) + "\n")
                    for r in sorted_indx[b]:
                        indx = b * self.test_rollouts + r
                        if rewards[indx] == self.positive_reward:
                            rev = 1
                        else:
                            rev = -1
                        answers.append(self.rev_entity_vocab[se[b,r]]+'\t'+ self.rev_entity_vocab[ce[b,r]]+'\t'+ str(self.log_probs[b,r])+'\n')
                        paths[str(qr)].append(
                            '\t'.join([str(self.rev_entity_vocab[e[indx]]) for e in
                                       self.entity_trajectory]) + '\n' + '\t'.join(
                                [str(self.rev_relation_vocab[re[indx]]) for re in self.relation_trajectory]) + '\n' + '\t'.join(
                                [str(self.rev_tim_vocab[te[indx]]) for te in self.tim_trajectory]) + '\n'+ str(
                                rev) + '\n' + str(
                                self.log_probs[b, r]) + '\n___' + '\n')
                    paths[str(qr)].append("#####################\n")


            all_final_reward_1 += final_reward_1
            all_final_reward_3 += final_reward_3
            all_final_reward_5 += final_reward_5
            all_final_reward_10 += final_reward_10
            all_final_reward_20 += final_reward_20
            auc += AP


        all_final_reward_1 /= total_examples
        all_final_reward_3 /= total_examples
        all_final_reward_5 /= total_examples
        all_final_reward_10 /= total_examples
        all_final_reward_20 /= total_examples
        auc /= total_examples

        if save_model:
            if all_final_reward_10 >= self.max_hits_at_10:
                self.max_hits_at_10 = all_final_reward_10
                self.save_path = self.model_saver.save(sess, self.model_dir + "model" + '.ckpt')

        if print_paths:
            logger.info("[ printing paths at {} ]".format(self.output_dir+'/test_beam/'))
            for q in paths:
                j = q.replace('/', '-')
                with codecs.open(self.path_logger_file_ + '_' + j, 'a', 'utf-8') as pos_file:
                    for p in paths[q]:
                        pos_file.write(p)
            with open(self.path_logger_file_ + 'answers', 'w') as answer_file:
                for a in answers:
                    answer_file.write(a)

        with open(self.output_dir + '/scores.txt', 'a') as score_file:
            score_file.write("Hits@1: {0:7.4f}".format(all_final_reward_1))
            score_file.write("\n")
            score_file.write("Hits@3: {0:7.4f}".format(all_final_reward_3))
            score_file.write("\n")
            score_file.write("Hits@5: {0:7.4f}".format(all_final_reward_5))
            score_file.write("\n")
            score_file.write("Hits@10: {0:7.4f}".format(all_final_reward_10))
            score_file.write("\n")
            score_file.write("Hits@20: {0:7.4f}".format(all_final_reward_20))
            score_file.write("\n")
            score_file.write("auc: {0:7.4f}".format(auc))
            score_file.write("\n")
            score_file.write("\n")

        logger.info("Hits@1: {0:7.4f}".format(all_final_reward_1))
        logger.info("Hits@3: {0:7.4f}".format(all_final_reward_3))
        logger.info("Hits@5: {0:7.4f}".format(all_final_reward_5))
        logger.info("Hits@10: {0:7.4f}".format(all_final_reward_10))
        logger.info("Hits@20: {0:7.4f}".format(all_final_reward_20))
        logger.info("auc: {0:7.4f}".format(auc))


    def top_k(self, scores, k):
        scores = scores.reshape(-1, k * self.max_num_actions)  # [B, (k*max_num_actions)]
        idx = np.argsort(scores, axis=1)
        idx = idx[:, -k:]  # take the last k highest indices # [B , k]
        return idx.reshape((-1))


    def update_params(optimizer, params, gradients, lr):
        ## 更新参数的通用函数，考虑梯度裁剪等优化策略
        optimizer.zero_grad()
        for param, grad in zip(params, gradients):
            param.grad = grad
            clip_grad_norm_(params, max_norm=1.0)  # 防止梯度爆炸
        optimizer.step()


    def meta_train(ftmi_model, meta_train_tasks, meta_lr, task_lr, num_iterations):
        # 初始化超参数和模型参数
        ftmi_model.train()
        meta_policy_params = list(ftmi_model.strategy_network.parameters())
        meta_optimizer = optim.Adam(meta_policy_params, lr=meta_lr)

        for iteration in range(num_iterations):
            # 打乱任务顺序以增加多样性
            random.shuffle(meta_train_tasks)

        for task in meta_train_tasks:
            # 为当前任务采样支持集和查询集
            support_set, query_set = task.sample_data()

            # 使用当前模型参数在任务上计算损失
            task_loss = loss_function(ftmi_model(support_set, query_set))

            # 计算针对当前任务的梯度并更新参数，模拟内在循环更新
            task_gradients = torch.autograd.grad(task_loss, ftmi_model.parameters(), create_graph=True)
            updated_task_params = [(param - task_lr * grad) for param, grad in
                                   zip(ftmi_model.parameters(), task_gradients)]

            # 使用更新后的参数在查询集上评估，获取新梯度用于外循环更新
            eval_loss = loss_function(ftmi_model(query_set, use_updated_params=True))  # 假设模型有切换参数机制
            meta_gradients = torch.autograd.grad(eval_loss, meta_policy_params)

            # 更新元策略网络参数
            update_params(meta_optimizer, meta_policy_params, meta_gradients, meta_lr)

            # 恢复模型参数至元学习前的状态，准备下一个任务
            ftmi_model.load_state_dict(saved_model_state)  # 假设之前保存了状态

    return ftmi_model

    def calculate_loss(self, support_set, query_set, task_rel_repr, pre_trained_tkg_embeddings):
        """
        计算损失函数，结合策略网络输出和强化学习奖励。
        """
        total_loss = 0
        for query in query_set:
            # 假设策略网络输出为一系列动作（关系和时间戳）
            actions = self.policy_network(task_rel_repr, query)

            # 计算路径得分，这里简化处理，实际应基于路径的合理性、新颖性等
            path_score = self.compute_path_score(actions, pre_trained_tkg_embeddings)

            # 设计一个奖励函数，平衡路径长度和新颖性
            reward = self.balance_novelty_and_length(actions)

            # 损失可以是负的奖励值，因为我们希望最大化奖励
            loss = -reward + (1 - path_score)  # 示例惩罚项，根据实际情况调整

            total_loss += loss

        avg_loss = total_loss / len(query_set)
        return avg_loss
# 初始化模型、任务集合等
ftmi = FTMIModel()
meta_train_tasks = [...]  # 元训练任务集
meta_learning_rate = 0.001
task_learning_rate = 0.01
num_training_iterations = 1000

# 开始元学习训练
trained_ftmi = meta_train(ftmi, meta_train_tasks, meta_learning_rate, task_learning_rate, num_training_iterations)
if __name__ == '__main__':
#在if __name__ == 'main': 下的代码只有在第一种情况下（即文件作为脚本直接执行）才会被执行，而import到其他脚本中是不会被执行的
    # read command line options
    options = read_options()
    # Set logging
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(options['log_file_name'], 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)
    # read the vocab files, it will be used by many classes hence global scope
    logger.info('reading vocab files...')
    options['relation_vocab'] = json.load(open(options['vocab_dir'] + '/relation_vocab.json'))
    options['entity_vocab'] = json.load(open(options['vocab_dir'] + '/entity_vocab.json'))
    options['tim_vocab'] = json.load(open(options['vocab_dir'] + '/tim_vocab.json'))
    logger.info('Reading mid to name map')
    mid_to_word = {}
    # with open('/iesl/canvas/rajarshi/data/RL-Path-RNN/FB15k-237/fb15k_names', 'r') as f:
    #     mid_to_word = json.load(f)
    logger.info('Done..')
    logger.info('Total number of entities {}'.format(len(options['entity_vocab'])))
    logger.info('Total number of relations {}'.format(len(options['relation_vocab'])))
    logger.info('Total number of tims {}'.format(len(options['tim_vocab'])))
    save_path = ''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.log_device_placement = False


    #Training
    if not options['load_model']:
        trainer = Trainer(options)
        with tf.Session(config=config) as sess:
            sess.run(trainer.initialize())
            trainer.initialize_pretrained_embeddings(sess=sess)

            trainer.train(sess)
            save_path = trainer.save_path
            path_logger_file = trainer.path_logger_file
            output_dir = trainer.output_dir

        tf.reset_default_graph()
    #Testing on test with best model
    else:
        logger.info("Skipping training")
        logger.info("Loading model from {}".format(options["model_load_dir"]))

    trainer = Trainer(options)
    if options['load_model']:
        save_path = options['model_load_dir']
        path_logger_file = trainer.path_logger_file
        output_dir = trainer.output_dir
    with tf.Session(config=config) as sess:

        trainer.initialize(restore=save_path, sess=sess)
        init = tf.global_variables_initializer()
        sess.run(init)

        trainer.test_rollouts = 100

        os.mkdir(path_logger_file + "/" + "test_beam")
        trainer.path_logger_file_ = path_logger_file + "/" + "test_beam" + "/paths"
        with open(output_dir + '/scores.txt', 'a') as score_file:
            score_file.write("Test (beam) scores with best model from " + str(save_path) + "\n")
        trainer.test_environment = trainer.test_test_environment
        trainer.test_environment.test_rollouts = 100

        trainer.test(sess, beam=False, print_paths=True, save_model=False)



        print ("options['nell_evaluation']")
        if options['nell_evaluation'] == 1:
            #nell_eval(path_logger_file + "/" + "test_beam/" + "pathsanswers", trainer.data_input_dir+'/sort_test.pairs' )
            nell_eval(path_logger_file + "/" + "test_beam/" + "pathsanswers", trainer.data_input_dir+'/sort_test.pairs' )
