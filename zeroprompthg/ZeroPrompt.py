

from .model import *
from .utils import *
from .downprompt import featureprompt
from .TxData import TxData


import sys
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(0)
#device = torch.device("cuda:0")

class ZeroPrompt:
    
    def __init__(self, data,
                       weight_bias_track = False,
                       device = 'cuda:0'):
        self.device = torch.device(device)
        self.weight_bias_track = weight_bias_track
        self.G = data.G
        self.df, self.df_train, self.df_valid, self.df_test = data.df, data.df_train, data.df_valid, data.df_test
        self.data_folder = data.data_folder
        self.disease_eval_idx = data.disease_eval_idx
        self.split = data.split
        self.no_kg = data.no_kg
        
        self.disease_rel_types = ['rev_contraindication', 'rev_indication', 'rev_off-label use']
        
        self.dd_etypes= [('drug', 'contraindication', 'disease'),
                  ('drug', 'indication', 'disease'),
                  ('drug', 'off-label use', 'disease'),
                  ('disease', 'rev_contraindication', 'drug'),
                  ('disease', 'rev_indication', 'drug'),
                  ('disease', 'rev_off-label use', 'drug')]


    def model_initialize(self, n_hid = 128,
                               n_inp = 128,
                               n_out = 128,
                               proto = True,
                               proto_num = 5,
                               sim_measure = 'all_nodes_profile',
                               bert_measure = 'disease_name',
                               agg_measure = 'rarity',
                               exp_lambda = 0.7,
                               num_walks = 200,
                               walk_mode = 'bit',
                               path_length = 2):

        if self.no_kg and proto:
            print('Ablation study on No-KG. No proto learning is used...')
            proto = False

        self.G = self.G.to('cpu')
        self.G = initialize_node_embedding(self.G, n_inp)
        self.g_valid_pos, self.g_valid_neg = evaluate_graph_construct(self.df_valid, self.G, 'fix_dst', 1, self.device)
        self.g_test_pos, self.g_test_neg = evaluate_graph_construct(self.df_test, self.G, 'fix_dst', 1, self.device)

        self.config = {'n_hid': n_hid,
                       'n_inp': n_inp,
                       'n_out': n_out,
                       'proto': proto,
                       'proto_num': proto_num,
                       'sim_measure': sim_measure,
                       'bert_measure': bert_measure,
                       'agg_measure': agg_measure,
                       'num_walks': num_walks,
                       'walk_mode': walk_mode,
                       'path_length': path_length
                      }

        self.model = HeteroRGCN(self.G,
                   in_size=n_inp,
                   hidden_size=n_hid,
                   out_size=n_out,
                   proto = proto,
                   proto_num = proto_num,
                   sim_measure = sim_measure,
                   bert_measure = bert_measure,
                   agg_measure = agg_measure,
                   num_walks = num_walks,
                   walk_mode = walk_mode,
                   path_length = path_length,
                   split = self.split,
                   data_folder = self.data_folder,
                   exp_lambda = exp_lambda,
                   device = self.device
                  ).to(self.device)



    def prompt(self, n_epoch = 500,
                       learning_rate = 1e-3,
                       train_print_per_n = 5,
                       valid_per_n = 25,
                       sweep_wandb = None,
                       save_name = None,
                 model_path = './',
                 save_result_path ='./'):

        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.device)
        self.G = self.G.to(self.device)
        neg_sampler = Full_Graph_NegSampler(self.G, 1, 'fix_dst', self.device)


        params = [param for key, param in self.model.prompt.items() if isinstance(param, nn.Parameter)]
        optimizer = torch.optim.AdamW(params + list(self.model.pred.parameters()), lr = learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.8)


        for epoch in range(n_epoch):

            negative_graph = neg_sampler(self.G)

            # pred_score_pos, pred_score_neg, pos_score, neg_score = self.model.forward_prompt(feature_prompt,self.G, negative_graph, pretrain_mode = False, mode = 'train')

            pred_score_pos, pred_score_neg, pos_score, neg_score = self.model(self.G, negative_graph, pretrain_mode=False,mode='train')

            pos_score = torch.cat([pred_score_pos[i] for i in self.dd_etypes])
            neg_score = torch.cat([pred_score_neg[i] for i in self.dd_etypes])

            scores = torch.sigmoid(torch.cat((pos_score, neg_score)).reshape(-1,))
            labels = [1] * len(pos_score) + [0] * len(neg_score)
            loss = F.binary_cross_entropy(scores, torch.Tensor(labels).float().to(self.device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step(loss)

            if self.weight_bias_track:
                self.wandb.log({"Training Loss": loss})

            if epoch % train_print_per_n == 0:
                # training tracking...
                auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc = get_all_metrics_fb(pred_score_pos, pred_score_neg, scores.reshape(-1,).detach().cpu().numpy(), labels, self.G, True)

                # if self.weight_bias_track:
                #     temp_d = get_wandb_log_dict(auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc, macro_auprc, "Training")
                #     temp_d.update({"LR": optimizer.param_groups[0]['lr']})
                #     self.wandb.log(temp_d)

                print('Epoch: %d LR: %.5f Loss %.4f, Train Micro AUROC %.4f Train Micro AUPRC %.4f Train Macro AUROC %.4f Train Macro AUPRC %.4f' % (
                    epoch,
                    optimizer.param_groups[0]['lr'],
                    loss.item(),
                    micro_auroc,
                    micro_auprc,
                    macro_auroc,
                    macro_auprc
                ))

                print('----- AUROC Performance in Each Relation -----')
                print_dict(auroc_rel)
                print('----- AUPRC Performance in Each Relation -----')
                print_dict(auprc_rel)
                print('----------------------------------------------')

            del pred_score_pos, pred_score_neg, scores, labels

            # Redirect stdout to the file
            print('Testing...')

            with torch.no_grad():
                (auroc_rel, auprc_rel, micro_auroc, micro_auprc, macro_auroc,
                 macro_auprc), loss, pred_pos, pred_neg = evaluate_fb(self.model, self.g_test_pos,
                                                                      self.g_test_neg, self.G, self.dd_etypes,
                                                                      self.device, True, mode='test')
            print(
                'Testing Loss %.4f Testing Micro AUROC %.4f Testing Micro AUPRC %.4f Testing Macro AUROC %.4f Testing Macro AUPRC %.4f' % (
                    loss,
                    micro_auroc,
                    micro_auprc,
                    macro_auroc,
                    macro_auprc
                ))

            print('----- AUROC Performance in Each Relation -----')
            print_dict(auroc_rel, dd_only=True)
            print('----- AUPRC Performance in Each Relation -----')
            print_dict(auprc_rel, dd_only=True)
            print('----------------------------------------------')


