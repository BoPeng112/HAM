'''
by chen ma
https://github.com/allenjack/HGN
'''

import torch
from utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_model(train_data, test_data, config, logger):
    num_users = train_data.num_users
    num_items = train_data.num_items

    # convert to sequences, targets and users
    sequences_np = train_data.sequences.sequences
    targets_np = train_data.sequences.targets
    users_np = train_data.sequences.user_ids

    train_matrix = train_data.tocsr()

    n_train = sequences_np.shape[0]
    logger.info("Total training records:{}".format(n_train))

    if config.model == 'xHAM':
        from model.xHAM import Model
    elif config.model == 'HGN':
        from model.HGN import Model

    ham = Model(num_users, num_items, config, device).to(device)

    optimizer = torch.optim.Adam(ham.parameters(), lr=config.learning_rate, weight_decay=config.l2)

    record_indexes = np.arange(n_train)
    batch_size = config.batch_size
    num_batches = int(n_train / batch_size) + 1
    for epoch_num in range(config.n_iter):

        t1 = time()

        # set model to training mode
        ham.train()

        np.random.shuffle(record_indexes)

        t_neg_start = time()
        negatives_np_multi = generate_negative_samples(train_matrix, config.neg_samples, config.sets_of_neg_samples)
        logger.info("Negative sampling time: {}s".format(time() - t_neg_start))

        epoch_loss = 0.0
        for batchID in range(num_batches):
            start = batchID * batch_size
            end = start + batch_size

            if batchID == num_batches - 1:
                if start < n_train:
                    end = n_train
                else:
                    break

            batch_record_index = record_indexes[start:end]

            batch_users = users_np[batch_record_index]
            batch_sequences = sequences_np[batch_record_index]
            batch_targets = targets_np[batch_record_index]
            negatives_np = negatives_np_multi[batchID % config.sets_of_neg_samples]
            batch_neg = negatives_np[batch_users]

            batch_users = torch.from_numpy(batch_users).type(torch.LongTensor).to(device)
            batch_sequences = torch.from_numpy(batch_sequences).type(torch.LongTensor).to(device)
            batch_targets = torch.from_numpy(batch_targets).type(torch.LongTensor).to(device)
            batch_negatives = torch.from_numpy(batch_neg).type(torch.LongTensor).to(device)

            items_to_predict = torch.cat((batch_targets, batch_negatives), 1)
            prediction_score = ham(batch_sequences, batch_users, items_to_predict, False)

            (targets_prediction, negatives_prediction) = torch.split(
                prediction_score, [batch_targets.size(1), batch_negatives.size(1)], dim=1)

            # compute the BPR loss
            loss = -torch.log(torch.sigmoid(targets_prediction - negatives_prediction) + 1e-8)
            loss = torch.mean(torch.sum(loss))

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss /= num_batches

        t2 = time()

        output_str = "Epoch %d [%.1f s]  loss=%.4f" % (epoch_num + 1, t2 - t1, epoch_loss)
        logger.info(output_str)

        if (epoch_num + 1) % 20 == 0:
            ham.eval()
            precision, recall, MAP, ndcg = evaluation(ham, train_data, test_data, config, device, topk=20)
            logger.info(', '.join(str(e) for e in precision))
            logger.info(', '.join(str(e) for e in recall))
            logger.info(', '.join(str(e) for e in MAP))
            logger.info(', '.join(str(e) for e in ndcg))
            logger.info("Evaluation time:{}".format(time() - t2))

            #the code below is used to save models
            #save_root ='all_models/'+config.model+'/'+config.setting+'/'+config.data+'/'+'model_'+str(epoch_num + 1)
            #torch.save(ham, save_root)

    logger.info("\n")
    logger.info("\n")


