#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import os.path

import numpy as np
import tensorflow as tf
import math

import input_data
import VGG
import tools

IMG_W = 32
IMG_H = 32
N_CLASSES = 10
BATCH_SIZE = 32
learning_rate = 0.01
MAX_STEP = 20000   # it took me about one hour to complete the training.
IS_PRETRAIN = True


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# %%
def train():
    
    pre_trained_weights = './vgg_pretrain/vgg16.npy'
    data_dir = './data/cifar-10-batches-bin/'
    train_log_dir = './logs/vgg16_logs/train/'
    val_log_dir = './logs/vgg16_logs/val/'
    
    with tf.name_scope('input'):
        tra_image_batch, tra_label_batch = input_data.read_cifar10(data_dir=data_dir,
                                                                   is_train=True,
                                                                   batch_size=BATCH_SIZE,
                                                                   shuffle=True)
        val_image_batch, val_label_batch = input_data.read_cifar10(data_dir=data_dir,
                                                                   is_train=False,
                                                                   batch_size= BATCH_SIZE,
                                                                   shuffle=False)
        
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int32, shape=[BATCH_SIZE, N_CLASSES]) 
    
    logits = VGG.VGG16N(x, N_CLASSES, IS_PRETRAIN)
    loss = tools.loss(logits, y_)
    accuracy = tools.accuracy(logits, y_)
    
    my_global_step = tf.Variable(0, name='global_step', trainable=False) 
    train_op = tools.optimize(loss, learning_rate, my_global_step)   
    
    saver = tf.train.Saver(tf.global_variables())  
    summary_op = tf.summary.merge_all()
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        
        # load the parameter file, assign the parameters, skip the specific layers
        tools.load_with_skip(pre_trained_weights, sess, ['fc6', 'fc7', 'fc8'])
    
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)
          
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                        break
                    
                tra_images, tra_labels = sess.run([tra_image_batch, tra_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                                feed_dict={x: tra_images, y_: tra_labels})
                if step % 50 == 0 or (step + 1) == MAX_STEP:                 
                    print ('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
                    train_summary_str = sess.run(summary_op,
                                                 feed_dict={x: tra_images, y_: tra_labels})
                    train_summary_writer.add_summary(train_summary_str, step)
                    
                if step % 200 == 0 or (step + 1) == MAX_STEP:
                    val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                    val_loss, val_acc = sess.run([loss, accuracy],
                                                 feed_dict={x: val_images, y_: val_labels})
                    print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' % (step, val_loss, val_acc))
                    val_summary_str = sess.run(summary_op,
                                               feed_dict={x: val_images, y_: val_labels})
                    val_summary_writer.add_summary(val_summary_str, step)
                        
                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
            
        coord.join(threads)


# %%
def evaluate():
    with tf.Graph().as_default():
        
        log_dir = './logs/vgg16_logs/train/'
        data_dir = './data/cifar-10-batches-bin/'
        n_test = 10000
                
        images, labels = input_data.read_cifar10(data_dir=data_dir,
                                                 is_train=False,
                                                 batch_size=BATCH_SIZE,
                                                 shuffle=False)

        logits = VGG.VGG16N(images, N_CLASSES, IS_PRETRAIN)  # shape of logits: [Batch_size, n_classes]
        correct = tools.num_correct_prediction(logits, labels)  # 得到准确率，类型是浮点型
        saver = tf.train.Saver(tf.global_variables())
        
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
                return
        
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            try:
                print('\nEvaluating......')
                # math.floor()返回小于或等于一个给定数字的最大整数
                num_step = int(math.floor(n_test / BATCH_SIZE))  # num_step = 312
                num_sample = num_step*BATCH_SIZE   # num_sample=9984
                step = 0
                total_correct = 0
                while step < num_step and not coord.should_stop():
                    batch_correct = sess.run(correct)    # 得到在一个batch中正确的个数
                    total_correct += np.sum(batch_correct)  # 得到总共的正确数量
                    step += 1
                print('Total testing samples: %d' % num_sample)
                print('Total correct predictions: %d' % total_correct)
                print('Average accuracy: %.2f%%' % (100*total_correct/num_sample))
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)


if __name__ == '__main__':
    evaluate()


