import seaborn as sns
import matplotlib.pyplot as plt
import torch.optim as optim
import visdom

from data.rai_ges_dataset import RAIGesDataSplitter
from data.real_time_dataset import get_real_time_data
from log_helper import LogHelper
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC, MulticlassAveragePrecision
from torchmetrics.functional import accuracy
from sklearn.metrics import confusion_matrix
from model.compare_methods import DiGesture, RadarNet, DeepSolid, Resnet50Classifier, MobilenetV350Classifier, RFDual
from data.mcd_dataset import *
from model.network import *
from data.cubelern_arm_dataset import *
from result_collector import cross_domain_results

# 0:'RANGE_ANGLE_IMAGE',
# 1:'TIME_FREQUENCY_IMAGE',
# 2:'COMPLEX_RANGE_DOPPLER',
# 3:'SINGLE_RANGE_DOPPLER'
RANGE_ANGLE_IMAGE = 'RANGE_ANGLE_IMAGE'
TIME_FREQUENCY_IMAGE = 'TIME_FREQUENCY_IMAGE'
COMPLEX_RANGE_DOPPLER = 'COMPLEX_RANGE_DOPPLER'
SINGLE_RANGE_DOPPLER = 'SINGLE_RANGE_DOPPLER'
CROPPED_RANGE_DOPPLER_IMAGER = 'CROPPED_RANGE_DOPPLER_IMAGER'


def seed_worker(worker_id):
    seed = worker_id + 1998
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def set_random_seed(seed=1998):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_confusion_matrix(best_ture_label, best_predict_label, title):
    cm = confusion_matrix(best_ture_label, best_predict_label)
    col_sum = np.sum(cm, axis=0)
    cm = np.round(100 * cm / col_sum[np.newaxis, :], 1)

    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig('cm_{}.png'.format(title))
    plt.show()
    plt.clf()


def deep_rai_collate_fn(datas_and_labels):
    datas_and_labels = sorted(datas_and_labels, key=lambda x: x[0].size()[0], reverse=True)
    datas = [item[0] for item in datas_and_labels]
    labels = torch.stack([item[2] for item in datas_and_labels])
    tracks = torch.stack([item[1] for item in datas_and_labels])
    data_lengths = [len(x) for x in datas]
    datas = pad_sequence(datas, batch_first=True, padding_value=0)
    return datas, torch.tensor(data_lengths), tracks, labels


def dynamic_sequence_collate_fn(datas_and_labels):
    datas_and_labels = sorted(datas_and_labels, key=lambda x: x[0].size()[0], reverse=True)
    datas = [item[0] for item in datas_and_labels]
    labels = torch.stack([item[-1] for item in datas_and_labels])
    data_lengths = [len(x) for x in datas]
    datas = pad_sequence(datas, batch_first=True, padding_value=0)
    return datas, torch.tensor(data_lengths), labels


def radar_net_unify_sequence(x):
    x = x.unsqueeze(0)
    x = F.interpolate(x, size=(x.size(-3), 24, 16), mode='trilinear', align_corners=False)
    x = torch.normal(mean=1, std=0.025, size=(1,)) * x
    return torch.squeeze(x)


def radar_net_collate_fn(datas_and_labels):
    datas_and_labels = sorted(datas_and_labels, key=lambda x: x[0].size()[0], reverse=True)
    datas = [radar_net_unify_sequence(item[0]) for item in datas_and_labels]
    labels = torch.stack([item[1] for item in datas_and_labels])
    data_lengths = [len(x) for x in datas]
    datas = pad_sequence(datas, batch_first=True, padding_value=0)
    return datas, torch.tensor(data_lengths), labels


def unpack_run_model(model, pack, device):
    labels = pack[-1].to(device)
    args = pack[:-1]
    for i, arg in enumerate(args):
        args[i] = args[i].to(device)
    outputs = model(*args)
    return outputs, labels


def get_correct_num(outputs, labels):
    prediction = torch.argmax(outputs, 1)
    return (prediction == labels).sum().float().item()


'''
def train(model, dataloader, epoch, optimizer, criterion, device):
    running_loss = 0.0
    total_sample = 0.0
    correct_sample = 0.0
    model.train()
    for i, pack in enumerate(dataloader):
        optimizer.zero_grad()
        outputs, labels = unpack_run_model(model, pack, device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # calculation of accuracy
        total_sample += labels.size(0)
        correct_sample += get_correct_num(outputs, labels)
        if i % 5 == 4:  # print every 5 mini-batches
            print('Training [%d, %5d] loss: %.3f, accuracy: %.5f' % (
                epoch + 1, i + 1, loss / (i + 1), correct_sample / total_sample))
    train_acc = correct_sample / total_sample
    print(
        '[Train] epoch:%d all_samples: %.5f, correct_samples: %.5f,  loss: %.5f, accuracy: %.5f' % (epoch,
                                                                                                    total_sample,
                                                                                                    correct_sample,
                                                                                                    running_loss / len(
                                                                                                        dataloader),
                                                                                                    train_acc))
    return train_acc, running_loss / len(dataloader)


def validate_or_test(model, dataloader, is_val=False, num_classes=7, criterion=None, auc_compute=None, ap_compute=None,
                     device=None):
    model.eval()
    running_loss = 0.0
    total_sample = 0.0
    correct_sample = 0.0
    ture_label = []
    predict_label = []
    with torch.no_grad():
        for i, pack in enumerate(dataloader):
            outputs, labels = unpack_run_model(model, pack, device)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            # calculation of accuracy
            total_sample += labels.size(0)
            correct_sample += get_correct_num(outputs, labels)
            prediction = torch.argmax(outputs, 1)
            predict_label.extend([x.item() for x in prediction])
            ture_label.extend([x.item() for x in labels])
            auc_compute.update(outputs, labels)
            ap_compute.update(outputs, labels)
    val_acc = correct_sample / total_sample
    val_auc = auc_compute.compute()

    val_micro_acc = accuracy(torch.tensor(predict_label), torch.tensor(ture_label), num_classes=num_classes,
                             average='macro', task="multiclass")
    val_ap = ap_compute.compute()
    ap_compute.reset()
    auc_compute.reset()
    if is_val:
        prefix = "[Validation]"
    else:
        prefix = "[Test]"
    print(prefix + ' all validation: %.5f, correct validation: %.5f' % (total_sample, correct_sample))
    print(prefix + ' val loss: %.5f, auc: %.5f, ap: %.5f, macro accuracy %.5f, accuracy: %.5f' % (
        running_loss / len(dataloader), val_auc, val_ap, val_micro_acc, val_acc))

    return val_acc, running_loss / len(dataloader), val_auc.item(), val_ap.item(), predict_label, ture_label


def train_and_val(model, train_set, val_set, start_epoch, total_epoch, batch_size, lr, title='', optimizer=None,
                  collate_fn=dynamic_sequence_collate_fn, model_name="test.pth", num_classes=7, criterion=None,
                  auc_compute=None, ap_compute=None, visual=False, device=None):
    best_acc = 0
    best_loss = 1e9
    best_auc = 0
    best_ap = 0
    best_ture_label = []
    best_predict_label = []

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8,
                                               pin_memory=True,
                                               worker_init_fn=seed_worker,
                                               collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                             num_workers=8,
                                             worker_init_fn=seed_worker,
                                             pin_memory=True,
                                             collate_fn=collate_fn)
    if start_epoch != 0:
        model.load_state_dict(torch.load('checkpoint/' + model_name)['model_state_dict'])
    model = model.to(device)
    # if vae is not None:
    #     vae = vae.to(device)
    if optimizer is None:
        optimizer = optim.Adam([
            {'params': model.parameters()},
        ], lr=lr[0])

    for epoch in range(start_epoch, total_epoch):
        if lr is not None:
            optimizer.param_groups[0]['lr'] = lr[epoch]

        train_acc, train_loss = train(model, train_loader, epoch, optimizer, criterion, device)
        val_acc, val_loss, val_auc, val_ap, pred_label, ture_label = validate_or_test(model, val_loader, is_val=True,
                                                                                      num_classes=num_classes,
                                                                                      auc_compute=auc_compute,
                                                                                      criterion=criterion,
                                                                                      device=device,
                                                                                      ap_compute=ap_compute)
        if visual:
            vis.line(X=np.array([epoch + 1]), Y=np.array([[train_acc, val_acc]]), win='acc',
                     update='append',
                     opts=dict(title='Train/val accuracy', xlabel='Epoch', ylabel='Accuracy',
                               legend=['Train accuracy', 'Val accuracy']))
            vis.line(X=np.array([epoch + 1]), Y=np.array([[train_loss, val_loss]]), win='loss',
                     update='append',
                     opts=dict(title='Train/val loss', xlabel='Epoch', ylabel='Loss',
                               legend=['Train loss', 'Val loss']))
            vis.line(X=np.array([epoch + 1]), Y=np.array([[val_auc, val_ap]]), win='ap',
                     update='append',
                     opts=dict(title='val auc/ap', xlabel='Epoch', ylabel='ap', legend=['Val auc', 'Val ap']))

        if val_loss < best_loss or (val_acc < best_acc and val_loss == best_loss):
            best_acc = val_acc
            best_loss = val_loss
            best_ap = val_ap
            best_auc = val_auc
            best_ture_label = ture_label
            best_predict_label = pred_label
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss
            }, 'checkpoint/' + model_name)
            torch.save({
                'model_state_dict': model.state_dict(),
            }, 'checkpoint/final_model.pth')
            print('saved')

        print('[Validation] best  auc: %.5f, ap: %.5f, accuracy: %.5f' % (
            best_auc, best_ap, best_acc))
    print(title + ' ' + str(best_acc))
    # precisions = compute_Precision(best_ture_label, best_predict_label)
    # loger.log('precision {}'.format(precisions))
    plot_confusion_matrix(best_ture_label, best_predict_label, title)

    return best_acc, best_auc, best_ap
'''


def get_model(model_type=0, out_size=6, data_type=RANGE_ANGLE_IMAGE, multistream=True, diff=True, attention=True,
              aug=True, dataset='', domain=0):
    collect_fn = None
    if model_type == 0:
        model = RAIRadarGestureClassifier(multistream=multistream, spatial_channels=(8, 16, 32),
                                          conv1d_channels=(8, 16, 32),
                                          heads=4,
                                          track_channels=(4, 8, 16), track_out_size=64, conv2d_feat_size=64,
                                          diff=diff, out_size=out_size,
                                          ra_feat_size=64, attention=attention,
                                          in_channel=1)
        collect_fn = deep_rai_collate_fn

    elif model_type == 1:
        model = RAIRadarGestureClassifier(multistream=multistream, spatial_channels=(8, 16, 32),
                                          conv1d_channels=(8, 16, 32),
                                          heads=4,
                                          track_channels=(4, 8, 16), track_out_size=32, conv2d_feat_size=32,
                                          diff=diff, out_size=out_size,
                                          ra_feat_size=32, attention=attention,
                                          in_channel=1)
        collect_fn = deep_rai_collate_fn
    elif model_type == 2:
        model = DiGesture(out_size=out_size)
        collect_fn = dynamic_sequence_collate_fn
    elif model_type == 3:
        model = RadarNet(out_size=out_size)
        collect_fn = radar_net_collate_fn
    elif model_type == 4:
        model = RFDual(out_size=out_size)
        collect_fn = dynamic_sequence_collate_fn
    elif model_type == 5:
        model = Resnet50Classifier(out_size=out_size)
    else:
        model = MobilenetV350Classifier(out_size=out_size)

    model_name = type(model).__name__
    if model_type == 1 or model_type == 0:

        if model_type == 1:
            model_name = model_name + '_small'
        if not multistream:
            model_name = model_name + '_single_stream'
        if not diff:
            model_name = model_name + '_no_diff'
        if not attention:
            model_name = model_name + '_no_attention'

        if data_type == SINGLE_RANGE_DOPPLER:
            model_name = model_name + '_rdi'
    if not aug:
        model_name += '_no_aug'
    model_name += '_' + dataset + str(domain)

    return model, model_name, collect_fn


def get_auc_ap_computer(num_classes=6, device=None):
    auc_cp = MulticlassAUROC(num_classes=num_classes, average='macro').to(device)
    ap_cp = MulticlassAveragePrecision(num_classes=num_classes, average='macro').to(device)
    return auc_cp, ap_cp


'''
def cube_k_fold(augmentation=True, epoch=200, start_epoch=0, domain=1, data_type=RANGE_ANGLE_IMAGE, batch_size=128,
                model_type=0,
                multistream=True, diff=True, attention=True, data_spliter=None, fold=None, reduction=False,
                device=None):
    set_random_seed()
    criterion = nn.CrossEntropyLoss()
    data_spliter.set_data_type(data_type)
    acc_auc_ap = np.zeros((1, data_spliter.get_domain_num(domain), 3))
    loger = LogHelper()

    for t in range(data_spliter.get_domain_num(domain)):
        train_index = []
        test_index = []
        for i in range(data_spliter.get_domain_num(domain)):
            if i != t:
                train_index.append(i)
            else:
                test_index.append(i)

        if reduction and t == data_spliter.get_domain_num(domain) - 1:
            continue

        train_set, test_set, _ = data_spliter.split_data(domain, train_index=train_index, test_index=test_index,
                                                         need_augmentation=augmentation, is_reduction=reduction)
        print('domain{} len{}', domain, train_set.len)
        if fold is not None and t < fold:
            continue
        model, model_name, collate_fn = get_model(model_type, data_spliter.get_class_num(), data_type, multistream,
                                                  diff, attention)

        print(type(model).__name__)
        lr_main = get_lr(epoch)

        acc_auc_ap[0, t, 0], acc_auc_ap[0, t, 1], acc_auc_ap[0, t, 2] = train_and_val(model, train_set, test_set,
                                                                                      start_epoch, epoch,
                                                                                      batch_size=batch_size,
                                                                                      lr=lr_main, title='in domain',
                                                                                      num_classes=data_spliter.get_class_num,
                                                                                      model_name=model_name,
                                                                                      collate_fn=collate_fn,
                                                                                      criterion=criterion,
                                                                                      device=device)

        loger.log(
            'k_fold_domain{}_complex{} model:{} dataset:{} augmentation:{} need_diff:{} multistream:{} acc:{} auc:{} ap:{}'
            .format(domain, t, type(model).__name__, type(data_spliter).__name__, augmentation, diff,
                    multistream, acc_auc_ap[0, t, 0], acc_auc_ap[0, t, 1], acc_auc_ap[0, t, 2]))
    cross_domain_results(model_name=model_name,
                         domain=domain,
                         train_indexes=[-1],
                         val_indexes=None,
                         test_indexes=np.arange(data_spliter.get_domain_num(domain)),
                         res=acc_auc_ap,
                         file_name='k_fold_result_{dataset}.xlsx'.format(dataset=type(data_spliter).__name__))


def cross_domain(augmentation=True, epoch=200, start_epoch=0, domain=1, data_type=RANGE_ANGLE_IMAGE, model_type=0,
                 batch_size=128,
                 train_index=None, test_index=None, need_test=False, val_time=5,
                 multistream=True, diff=True, attention=True, data_spliter=None, device=None):
    # lr_main = get_propsed_lr(augmentation, epoch)
    # complex_clear_cache()
    set_random_seed()
    data_spliter.set_data_type(data_type=data_type)
    criterion = nn.CrossEntropyLoss()
    if need_test:
        start = 1
    else:
        start = 0
    acc_auc_ap = np.zeros((val_time, len(test_index) + start, 3))
    auc_cp, ap_cp = get_auc_ap_computer(data_spliter.get_class_num(), device=device)
    for v_i in range(val_time):
        # domain, train_index, val_index, test_index, need_val, need_test, need_augmentation, is_reduction
        train_set, test_set, val_set = data_spliter.split_data(domain, train_index, None, None, need_val=True,
                                                               need_test=need_test, need_augmentation=augmentation)
        model, model_name, collate_fn = get_model(model_type, data_spliter.get_class_num(), data_type, multistream,
                                                  diff, attention)
        print('model:{} domain:{} len:{}', model_name, domain, train_set.len)
        lr_main = get_lr(epoch=epoch)
        train_and_val(model, train_set, val_set, start_epoch, epoch, batch_size=batch_size, lr=lr_main,
                      title='cross domain', model_name=model_name, collate_fn=collate_fn, auc_compute=auc_cp,
                      num_classes=data_spliter.get_class_num(), ap_compute=ap_cp, device=device, criterion=criterion)
        if need_test:
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                                      num_workers=8,
                                                      worker_init_fn=seed_worker,
                                                      pin_memory=True,
                                                      collate_fn=collate_fn)
            acc_auc_ap[v_i, 0, 0], test_loss, acc_auc_ap[v_i, 0, 1], acc_auc_ap[
                v_i, 0, 2], pred_label, ture_label = validate_or_test(model, test_loader, is_val=False,
                                                                      num_classes=data_spliter.get_class_num(),
                                                                      criterion=criterion,
                                                                      auc_compute=auc_cp, ap_compute=ap_cp,
                                                                      device=device)
        for i, t_i in enumerate(test_index, start=start):
            test_set = data_spliter.get_dataset([t_i])
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                                      num_workers=8,
                                                      worker_init_fn=seed_worker,
                                                      pin_memory=True,
                                                      collate_fn=collate_fn)
            acc_auc_ap[v_i, i, 0], test_loss, acc_auc_ap[v_i, i, 1], acc_auc_ap[
                v_i, i, 2], pred_label, ture_label = validate_or_test(model, test_loader, is_val=False,
                                                                      criterion=criterion,
                                                                      num_classes=data_spliter.get_class_num(),
                                                                      auc_compute=auc_cp, ap_compute=ap_cp,
                                                                      device=device)
    if need_test:
        test_index = [-1] + test_index
    cross_domain_results(model_name=model_name,
                         domain=domain,
                         train_indexes=train_index,
                         val_indexes=None,
                         test_indexes=test_index,
                         res=acc_auc_ap,
                         file_name='cross_domain_result_{dataset}.xlsx'.format(dataset=type(data_spliter).__name__))

    acc_auc_ap = np.mean(acc_auc_ap, axis=0)
    loger = LogHelper()
    for i, t_i in enumerate(test_index):
        loger.log(
            'cross_domain:{} train:{} test:{} model:{} dataset:{} augmentation:{} need_diff:{} multistream:{} acc:{} auc:{} ap:{}'
            .format(domain, train_index, t_i, model_name, type(data_spliter).__name__,
                    augmentation, diff, multistream, acc_auc_ap[i, 0], acc_auc_ap[i, 1], acc_auc_ap[i, 2]))

def train_transferring(augmentation=True, domain=1, model_type=0, data_type=RANGE_ANGLE_IMAGE, batch_size=128,
                       train_index=None, test_index=None, need_test=False, val_time=5,
                       pre_train_lr=0.001, train_lr=0.0001, pre_train_epoch=100, train_epoch=100,
                       multistream=True, diff=True, attention=True, device=None):
    mcd_data_splitter = MCDDataSplitter()
    rai_ges_splitter = RAIGesDataSplitter()
    criterion = nn.CrossEntropyLoss()
    set_random_seed()
    loger = LogHelper()
    pre_train_model, model_name, collate_fn = get_model(model_type, mcd_data_splitter.get_class_num(), data_type,
                                                        multistream, diff, attention)
    pre_trained_model_name = 'pre_trained_{name}.pth'.format(name=model_name)

    print("===============pre-training start!!!! model:{}==================".format(pre_trained_model_name))
    auc_cp, ap_cp = get_auc_ap_computer(mcd_data_splitter.get_class_num(), device=device)
    if not os.path.exists('checkpoint/' + pre_trained_model_name):
        train_set, test_set, _ = mcd_data_splitter.split_data(0, train_index=[0, 1, 2, 3], test_index=[4],
                                                              need_augmentation=augmentation)
        train_and_val(pre_train_model, train_set, test_set, 0, pre_train_epoch, batch_size=batch_size,
                      lr=[pre_train_lr] * pre_train_epoch, title='pre_train', criterion=criterion,
                      model_name=pre_trained_model_name, num_classes=mcd_data_splitter.get_class_num(),
                      collate_fn=collate_fn, auc_compute=auc_cp, ap_compute=ap_cp, device=device)
    print("===============pre-training completed!!!!==================")

    auc_cp, ap_cp = get_auc_ap_computer(rai_ges_splitter.get_class_num(), device=device)
    final_model_name = 'transferring_domain{domain}_model.pth'.format(domain=domain)
    # linear_in = pre_train_model.classifier.weight.shape[1]
    # pre_train_model.to('cpu')
    # pre_train_model.classifier = nn.Linear(linear_in, rai_ges_splitter.get_class_num())
    if need_test:
        start = 1
    else:
        start = 0
    acc_auc_ap = np.zeros((val_time, len(test_index) + start, 3))

    for v_i in range(val_time):
        train_model, model_name, collate_fn = get_model(model_type, rai_ges_splitter.get_class_num(), data_type,
                                                        multistream, diff, attention)
        params = torch.load('checkpoint/' + pre_trained_model_name)['model_state_dict']
        del params['classifier.weight']
        del params['classifier.bias']
        train_model.load_state_dict(params, strict=False)
        optimizer = optim.Adam([
            {'params': train_model.frame_model.parameters(), 'lr': 0.0001},
            {'params': train_model.temporal_model.parameters(), 'lr': 0.0003},
            {'params': train_model.tn.parameters(), 'lr': 0.0003},
            {'params': train_model.classifier.parameters(), 'lr': 0.001},
        ])
        train_set, test_set, val_set = rai_ges_splitter.split_data(domain, train_index=train_index, need_val=True,
                                                                   need_test=need_test,
                                                                   need_augmentation=augmentation)
        train_and_val(train_model, train_set, val_set, 0, train_epoch, batch_size=batch_size, criterion=criterion,
                      lr=None, title='transferring', optimizer=optimizer, model_name=final_model_name, device=device,
                      collate_fn=collate_fn, auc_compute=auc_cp, ap_compute=ap_cp,
                      num_classes=rai_ges_splitter.get_class_num())
        if need_test:
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                                      num_workers=8,
                                                      worker_init_fn=seed_worker,
                                                      pin_memory=True,
                                                      collate_fn=collate_fn)
            acc_auc_ap[v_i, 0, 0], test_loss, acc_auc_ap[v_i, 0, 1], acc_auc_ap[
                v_i, 0, 2], pred_label, ture_label = validate_or_test(train_model, test_loader, is_val=False,
                                                                      num_classes=rai_ges_splitter.get_class_num(),
                                                                      criterion=criterion,
                                                                      auc_compute=auc_cp, ap_compute=ap_cp,
                                                                      device=device)
        for i, t_i in enumerate(test_index, start=start):
            test_set = rai_ges_splitter.get_dataset([t_i])
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                                      num_workers=8,
                                                      worker_init_fn=seed_worker,
                                                      pin_memory=True,
                                                      collate_fn=collate_fn)
            acc_auc_ap[v_i, i, 0], test_loss, acc_auc_ap[v_i, i, 1], acc_auc_ap[
                v_i, i, 2], pred_label, ture_label = validate_or_test(train_model, test_loader, is_val=False,
                                                                      num_classes=rai_ges_splitter.get_class_num(),
                                                                      criterion=criterion,
                                                                      auc_compute=auc_cp, ap_compute=ap_cp,
                                                                      device=device)
    if need_test:
        test_index = [-1] + test_index
    cross_domain_results(model_name=final_model_name,
                         domain=domain,
                         train_indexes=train_index,
                         val_indexes=None,
                         test_indexes=test_index,
                         res=acc_auc_ap,
                         file_name='transferring_result.xlsx')

    acc_auc_ap = np.mean(acc_auc_ap, axis=0)

    for i, t_i in enumerate(test_index):
        loger.log(
            'transferring:{} train:{} test:{} model:{} augmentation:{} need_diff:{} multistream:{} acc:{} auc:{} ap:{}'
            .format(domain, train_index, t_i, model_name, augmentation, diff, multistream, acc_auc_ap[i, 0],
                    acc_auc_ap[i, 1], acc_auc_ap[i, 2]))

'''


def train_for_real_time(model_type=0, out_size=7, train_manager=None, epoch=200, batch_size=128):
    set_random_seed()

    model, model_name, collate_fn = get_model(model_type, out_size=out_size)
    if train_manager is None:
        train_manager = ModelTrainingManager(class_num=out_size)

    train_set, test_set = get_real_time_data()
    train_loader = get_dataloader(train_set, True, batch_size, collate_fn)
    test_loader = get_dataloader(test_set, False, batch_size, collate_fn)
    lr_main = get_lr(epoch)
    model_name = 'real_time_{}.pth'.format(model_name)
    acc, auc, ap = train_manager.train_and_val(model, train_loader, test_loader, epoch, lr_list=[lr_main],
                                               model_name=model_name)
    train_manager.loger.log('real_time ====== acc:{} auc:{} ap:{}'.format(acc, auc, ap))


def get_lr(epoch=200):
    lr_list = np.zeros(epoch)
    lr_list[:epoch // 2] = 0.001
    lr_list[epoch // 2:epoch // 2 + epoch // 4] = 0.0003
    lr_list[epoch // 2 + epoch // 4:] = 0.0001
    return lr_list


def get_dataloader(data_set, shuffle, batch_size, collate_fn):
    loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=shuffle,
                                         num_workers=8,
                                         worker_init_fn=seed_worker,
                                         pin_memory=True,
                                         collate_fn=collate_fn)
    return loader


def new_cube_k_fold(augmentation=True, epoch=200, start_epoch=0, domain=1, data_type=RANGE_ANGLE_IMAGE, batch_size=128,
                    model_type=0, train_manager=None,
                    multistream=True, diff=True, attention=True, data_spliter=None, fold=None, reduction=False, ):
    set_random_seed()
    data_spliter.set_data_type(data_type)
    acc_auc_ap = np.zeros((1, data_spliter.get_domain_num(domain), 3))
    if train_manager is None:
        train_manager = ModelTrainingManager(class_num=data_spliter.get_class_num())
    for t in range(data_spliter.get_domain_num(domain)):
        train_index = []
        test_index = []
        for i in range(data_spliter.get_domain_num(domain)):
            if i != t:
                train_index.append(i)
            else:
                test_index.append(i)

        if reduction and t == data_spliter.get_domain_num(domain) - 1:
            continue

        train_set, test_set, _ = data_spliter.split_data(domain, train_index=train_index, test_index=test_index,
                                                         need_augmentation=augmentation, is_reduction=reduction)
        print('domain{} len{}'.format(domain, train_set.len))

        if fold is not None and t < fold:
            continue

        model, model_name, collate_fn = get_model(model_type, data_spliter.get_class_num(), data_type, multistream,
                                                  diff, attention, dataset=type(data_spliter).__name__
                                                  , domain=domain)
        train_loader = get_dataloader(train_set, True, batch_size, collate_fn)
        test_loader = get_dataloader(test_set, False, batch_size, collate_fn)

        print(type(model).__name__)
        acc_auc_ap[0, t, 0], acc_auc_ap[0, t, 1], acc_auc_ap[0, t, 2] = \
            train_manager.train_and_val(model, train_loader, test_loader, epoch, lr_list=[get_lr(epoch)],
                                        model_name=model_name)

        train_manager.loger.log(
            'k_fold_domain{}_complex{} model:{} dataset:{} augmentation:{} need_diff:{} multistream:{} acc:{} auc:{} ap:{}'
            .format(domain, t, type(model).__name__, type(data_spliter).__name__, augmentation, diff,
                    multistream, acc_auc_ap[0, t, 0], acc_auc_ap[0, t, 1], acc_auc_ap[0, t, 2]))
    cross_domain_results(model_name=model_name,
                         domain=domain,
                         train_indexes=[-1],
                         val_indexes=None,
                         test_indexes=np.arange(data_spliter.get_domain_num(domain)),
                         res=acc_auc_ap,
                         file_name='k_fold_result_{dataset}.xlsx'.format(dataset=type(data_spliter).__name__))


def new_cross_domain(augmentation=True, epoch=200, start_epoch=0, domain=1, data_type=RANGE_ANGLE_IMAGE, model_type=0,
                     batch_size=128, train_index=None, test_index=None, need_test=False, val_time=5,
                     multistream=True, diff=True, attention=True, data_spliter=None, train_manager=None, need_save_result=True):
    if train_manager is None:
        train_manager = ModelTrainingManager(class_num=data_spliter.get_class_num())
    set_random_seed()
    data_spliter.set_data_type(data_type=data_type)
    if need_test:
        start = 1
    else:
        start = 0
    acc_auc_ap = np.zeros((val_time, len(test_index) + start, 3))
    for v_i in range(val_time):
        # domain, train_index, val_index, test_index, need_val, need_test, need_augmentation, is_reduction
        train_set, test_set, val_set = data_spliter.split_data(domain, train_index, None, None, need_val=True,
                                                               need_test=need_test, need_augmentation=augmentation)
        model, model_name, collate_fn = get_model(model_type, data_spliter.get_class_num(), data_type, multistream,
                                                  diff, attention, aug=augmentation, dataset=type(data_spliter).__name__
                                                  , domain=domain)
        train_loader = get_dataloader(train_set, True, batch_size, collate_fn)
        val_loader = get_dataloader(val_set, False, batch_size, collate_fn)
        print('model:{} domain:{} len:{}'.format(model_name, domain, train_set.len))
        train_manager.train_and_val(model, train_loader, val_loader, epoch, lr_list=[get_lr(epoch)],
                                    model_name=model_name)
        if need_test:
            test_loader = get_dataloader(test_set, False, batch_size, collate_fn)
            acc_auc_ap[v_i, 0, 0], _, acc_auc_ap[v_i, 0, 1], acc_auc_ap[
                v_i, 0, 2], _, _ = train_manager.test_or_val(model, test_loader)
        for i, t_i in enumerate(test_index, start=start):
            test_set = data_spliter.get_dataset([t_i])
            test_loader = get_dataloader(test_set, False, batch_size, collate_fn)
            acc_auc_ap[v_i, i, 0], _, acc_auc_ap[v_i, i, 1], acc_auc_ap[
                v_i, i, 2], _, _ = train_manager.test_or_val(model, test_loader)

    test_index = [-1] + test_index if need_test else test_index
    if need_save_result:
        cross_domain_results(model_name=model_name,
                             domain=domain,
                             train_indexes=train_index,
                             val_indexes=None,
                             test_indexes=test_index,
                             res=acc_auc_ap,
                             file_name='cross_domain_result_{dataset}.xlsx'.format(dataset=type(data_spliter).__name__))

    acc_auc_ap = np.mean(acc_auc_ap, axis=0)
    for i, t_i in enumerate(test_index):
        train_manager.loger.log(
            'cross_domain:{} train:{} test:{} model:{} dataset:{} augmentation:{} need_diff:{} multistream:{} acc:{} auc:{} ap:{}'
            .format(domain, train_index, t_i, model_name, type(data_spliter).__name__,
                    augmentation, diff, multistream, acc_auc_ap[i, 0], acc_auc_ap[i, 1], acc_auc_ap[i, 2]))


def get_trans_lr(epoch=200):
    lr_list = np.zeros((4, epoch))
    lr_list[0, :epoch // 2] = 0.0003
    lr_list[1, :epoch // 2] = 0.0003
    lr_list[2, :epoch // 2] = 0.001
    lr_list[3, :epoch // 2] = 0.001
    lr_list[0, epoch // 2:] = 0.0003
    lr_list[1, epoch // 2:] = 0.0003
    lr_list[2, epoch // 2:] = 0.001
    lr_list[3, epoch // 2:] = 0.001
    return lr_list


def new_train_transferring(augmentation=True, domain=1, model_type=0, data_type=RANGE_ANGLE_IMAGE, batch_size=128,
                           train_index=None, test_index=None, need_test=False, val_time=5, train_manager=None,
                           pre_train_lr=0.001, pre_train_epoch=100, train_epoch=100,
                           multistream=True, diff=True, attention=True):
    mcd_data_splitter = MCDDataSplitter(is_multi_negative=False)
    rai_ges_splitter = RAIGesDataSplitter()
    if train_manager is None:
        train_manager = ModelTrainingManager(class_num=mcd_data_splitter.get_class_num())
    set_random_seed()
    pre_train_model, model_name, collate_fn = get_model(model_type, mcd_data_splitter.get_class_num(), data_type,
                                                        multistream, diff, attention,
                                                        dataset=type(mcd_data_splitter).__name__, domain=0)
    pre_trained_model_name = 'pre_trained_{name}.pth'.format(name=model_name)
    print("===============pre-training start!!!! model:{}==================".format(pre_trained_model_name))

    if not os.path.exists('checkpoint/' + pre_trained_model_name):
        train_set, test_set, _ = mcd_data_splitter.split_data(0, train_index=[0, 1, 2, 3], test_index=[4],
                                                              need_augmentation=augmentation)
        train_loader = get_dataloader(train_set, True, batch_size, collate_fn)
        test_loader = get_dataloader(test_set, False, batch_size, collate_fn)
        train_manager.train_and_val(pre_train_model, train_loader, test_loader, pre_train_epoch,
                                    lr_list=[[pre_train_lr] * pre_train_epoch],
                                    model_name=pre_trained_model_name)
    print("===============pre-training completed!!!!==================")
    # load pre_trained model
    params = torch.load(os.path.join(train_manager.checkpoint_path, pre_trained_model_name))['model_state_dict']
    del params['classifier.weight']
    del params['classifier.bias']

    if need_test:
        start = 1
    else:
        start = 0
    acc_auc_ap = np.zeros((val_time, len(test_index) + start, 3))
    for v_i in range(val_time):
        train_model, model_name, collate_fn = get_model(model_type, rai_ges_splitter.get_class_num(), data_type,
                                                        multistream, diff, attention,
                                                        dataset=type(rai_ges_splitter).__name__, domain=0)
        final_model_name = 'transferring_d{domain}_{model_name}.pth'.format(domain=domain, model_name=model_name)
        train_model.load_state_dict(params, strict=False)
        optimizer = optim.Adam([
            {'params': train_model.frame_model.parameters(), 'lr': 0.0001},
            {'params': train_model.temporal_model.parameters(), 'lr': 0.0003},
            {'params': train_model.tn.parameters(), 'lr': 0.001},
            {'params': train_model.classifier.parameters(), 'lr': 0.001},
        ], lr=0.001)
        train_manager.set_optimizer(optimizer=optimizer)
        train_manager.set_class_num(rai_ges_splitter.get_class_num())
        train_set, test_set, val_set = rai_ges_splitter.split_data(domain, train_index=train_index, need_val=True,
                                                                   need_test=need_test,
                                                                   need_augmentation=augmentation)
        train_loader = get_dataloader(train_set, True, batch_size, collate_fn)
        val_loader = get_dataloader(val_set, False, batch_size, collate_fn)

        train_manager.train_and_val(train_model, train_loader, val_loader, train_epoch, lr_list=get_trans_lr(),
                                    model_name=final_model_name)

        print("[Results] transferring domain{}".format(domain))
        if need_test:
            test_loader = get_dataloader(test_set, False, batch_size, collate_fn)
            acc_auc_ap[v_i, 0, 0], _, acc_auc_ap[v_i, 0, 1], acc_auc_ap[v_i, 0, 2], _, _ = train_manager.test_or_val(
                train_model, test_loader)

        for i, t_i in enumerate(test_index, start=start):
            test_set = rai_ges_splitter.get_dataset([t_i])
            test_loader = get_dataloader(test_set, False, batch_size, collate_fn)
            acc_auc_ap[v_i, i, 0], _, acc_auc_ap[v_i, i, 1], acc_auc_ap[
                v_i, i, 2], _, _ = train_manager.test_or_val(train_model, test_loader)

    if need_test:
        test_index = [-1] + test_index
    cross_domain_results(model_name=final_model_name,
                         domain=domain,
                         train_indexes=train_index,
                         val_indexes=None,
                         test_indexes=test_index,
                         res=acc_auc_ap,
                         file_name='transferring_result.xlsx')

    acc_auc_ap = np.mean(acc_auc_ap, axis=0)

    for i, t_i in enumerate(test_index):
        train_manager.loger.log(
            'transferring:{} train:{} test:{} model:{} augmentation:{} need_diff:{} multistream:{} acc:{} auc:{} ap:{}'
            .format(domain, train_index, t_i, model_name, augmentation, diff, multistream, acc_auc_ap[i, 0],
                    acc_auc_ap[i, 1], acc_auc_ap[i, 2]))


class ModelTrainingManager:
    def __init__(self, device=None, criterion=None, optimizer=None, class_num=6, auc_compute=None,
                 ap_compute=None, need_print_result=True, print_step=5, visualize=False, vis_port=6006,
                 need_confusion_matrix=True, checkpoint_path='checkpoint'):
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.loger = LogHelper()
        self.need_print_result = need_print_result
        self.print_step = print_step
        self.ap_compute = ap_compute
        self.auc_compute = auc_compute
        self.class_num = class_num
        self.visualize = visualize
        self.checkpoint_path = checkpoint_path
        self.need_confusion_matrix = need_confusion_matrix
        self.trained_epoch = 0

        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        if device is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.visualize:
            self.vis = visdom.Visdom(env='model result', port=vis_port)
        if auc_compute is None or ap_compute is None:
            self.auc_compute, self.ap_compute = get_auc_ap_computer(class_num)

    def set_class_num(self, n):
        self.class_num = n
        self.auc_compute, self.ap_compute = get_auc_ap_computer(n)

    def set_optimizer(self, optimizer=None, model=None):
        self.trained_epoch = 0
        if optimizer is None and model is not None:
            self.optimizer = optim.Adam([
                {'params': model.parameters()},
            ], lr=0.001)
        else:
            self.optimizer = optimizer

    def print_result(self, acc=0.0, auc=0.0, ap=0.0, loss=0.0, samples=0, correct_samples=0, is_training=False,
                     epoch=None, batch_index=None):
        if not self.need_print_result or (batch_index is not None and not is_training):
            return
        if batch_index is None:
            prefix = '[Train     ] ' if is_training else '[Validation] '
            msg = prefix + 'epoch:{epoch} loss:{loss:.5f} auc:{auc:.4f} ap:{ap:.4f} acc:{acc:.4f}'.format(
                epoch=epoch,
                loss=loss,
                auc=auc,
                ap=ap,
                acc=acc,
            ) + ' samples:{n} cor_samples:{cn}'.format(n=samples, cn=correct_samples)
            print(msg)
        elif batch_index % self.print_step == 0:
            msg = 'Training [{epoch}, {t}] loss:{loss:.4f} acc:{acc:.4f}'.format(
                epoch=epoch,
                t=batch_index,
                loss=loss,
                acc=acc,
            )
            print(msg)

    def visualize_curve(self, epoch, train_acc, train_loss, val_acc, val_loss, val_auc, val_ap):
        self.vis.line(X=np.array([epoch + 1]), Y=np.array([[train_acc, val_acc]]), win='acc',
                      update='append',
                      opts=dict(title='Train/val accuracy', xlabel='Epoch', ylabel='Accuracy',
                                legend=['Train accuracy', 'Val accuracy']))
        self.vis.line(X=np.array([epoch + 1]), Y=np.array([[train_loss, val_loss]]), win='loss',
                      update='append',
                      opts=dict(title='Train/val loss', xlabel='Epoch', ylabel='Loss',
                                legend=['Train loss', 'Val loss']))
        self.vis.line(X=np.array([epoch + 1]), Y=np.array([[val_auc, val_ap]]), win='ap',
                      update='append',
                      opts=dict(title='val auc/ap', xlabel='Epoch', ylabel='ap', legend=['Val auc', 'Val ap']))

    def checkpoint(self, epoch, model, model_name, val_loss):
        if not model_name.endswith('.pth'):
            model_name += '.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': val_loss
        }, os.path.join(self.checkpoint_path, model_name))
        # torch.save({
        #    'model_state_dict': model.state_dict(),
        # }, os.path.join(self.checkpoint_path, 'final_model.pth'))
        print(
            '[Saved] model:{model_name} saved to fold:{path}'.format(model_name=model_name, path=self.checkpoint_path))

    def load_model(self, model, model_name):
        model.load_state_dict(torch.load(os.path.join(self.checkpoint_path, model_name))['model_state_dict'])

    def run_one_epoch(self, model, dataloader, epoch):
        running_loss = 0.0
        total_sample = 0.0
        correct_sample = 0.0
        ture_label = []
        predict_label = []
        for i, pack in enumerate(dataloader):
            if model.training:
                self.optimizer.zero_grad()
            outputs, labels = unpack_run_model(model, pack, self.device)
            loss = self.criterion(outputs, labels)
            if model.training:
                loss.backward()
                self.optimizer.step()
            running_loss += loss.item()
            prediction = torch.argmax(outputs.detach(), 1)
            if not model.training:
                predict_label.extend([x.item() for x in prediction])
                ture_label.extend([x.item() for x in labels])
                self.auc_compute.update(outputs.detach(), labels)
                self.ap_compute.update(outputs.detach(), labels)
            # return (prediction == labels).sum().float().item()
            # calculation of accuracy
            total_sample += labels.size(0)
            correct_sample += (prediction == labels).sum().float().item()
            self.print_result(epoch=epoch, batch_index=i, loss=running_loss, acc=correct_sample / total_sample,
                              is_training=model.training)
        train_acc = correct_sample / total_sample
        running_loss = running_loss / len(dataloader)
        if not model.training:
            auc = self.auc_compute.compute().item()
            ap = self.ap_compute.compute().item()
            self.auc_compute.reset()
            self.ap_compute.reset()
        else:
            auc = 0
            ap = 0
        self.print_result(epoch=epoch, acc=train_acc, auc=auc, ap=ap, loss=running_loss, is_training=model.training,
                          correct_samples=correct_sample, samples=total_sample)
        return train_acc, auc, ap, running_loss, predict_label, ture_label

    def train(self, model, dataloader, epoch):
        self.trained_epoch += 1
        model.train()
        acc, _, _, loss, _, _ = self.run_one_epoch(model, dataloader, epoch)
        return acc, loss

    def test_or_val(self, model, dataloader, epoch=-1):
        model.eval()
        acc, auc, ap, loss, pred_label, ture_label = self.run_one_epoch(model, dataloader, epoch)
        return acc, loss, auc, ap, pred_label, ture_label

    def train_and_val(self, model, train_dataloader, test_dataloader, total_epoch=200, start_epoch=0, lr_list=None,
                      model_name='test.pth', cm_title='test'):
        best_acc = 0
        best_loss = 1e9
        best_auc = 0
        best_ap = 0
        best_ture_label = []
        best_predict_label = []
        model = model.to(self.device)
        if self.trained_epoch != 0 or self.optimizer is None:
            self.set_optimizer(model=model)
        for epoch in range(start_epoch, total_epoch):
            if lr_list is not None:
                for i, lr in enumerate(lr_list):
                    self.optimizer.param_groups[i]['lr'] = lr[epoch]

            train_acc, train_loss = self.train(model, train_dataloader, epoch)
            val_acc, val_loss, val_auc, val_ap, pred_label, ture_label = self.test_or_val(model, test_dataloader, epoch)
            if self.visualize:
                self.visualize_curve(epoch, train_acc, train_loss, val_acc, val_loss, val_auc, val_ap)

            if val_loss < best_loss or (val_acc < best_acc and val_loss == best_loss):
                best_acc = val_acc
                best_loss = val_loss
                best_ap = val_ap
                best_auc = val_auc
                best_ture_label = ture_label
                best_predict_label = pred_label
                self.checkpoint(epoch, model, model_name, val_loss)
            print('[Current Best] auc: %.5f, ap: %.5f, acc: %.5f' % (best_auc, best_ap, best_acc))
        print( '[Training Finished] model:{model_name} auc:{auc:.5f}, ap:{ap:.5f}, acc:{acc:.5f}\n'.format(
                model_name=model_name,
                auc=best_auc,
                ap=best_ap,
                acc=best_acc))
        if self.need_confusion_matrix:
            plot_confusion_matrix(best_ture_label, best_predict_label, cm_title)

        return best_acc, best_auc, best_ap


if __name__ == '__main__':
    # loger = LogHelper()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # criterion = nn.CrossEntropyLoss()
    criterion_recon = nn.BCELoss()

    # ges_label = ['1', '2', '3', '4', '5', '6', '7']
    # ges_label = cube_gestures
    # num_class = 7
    # multiclassAUROC = MulticlassAUROC(num_classes=num_class, average='macro').to(device)
    # multiclassAveragePrecision = MulticlassAveragePrecision(num_classes=num_class, average='macro').to(device)

    di_DataSplitter = MCDDataSplitter()
    complex_DataSplitter = RAIGesDataSplitter()

    '''
    new_cross_domain(augmentation=True, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     # train_index=[0, 1, 2, 3, 4],
                     # test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     train_index=[0],
                     test_index=[1, 2, 3, 4, 5],
                     val_time=5,
                     # need_test=True,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=di_DataSplitter)

    new_cross_domain(augmentation=True, model_type=1, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     train_index=[0],
                     test_index=[1, 2, 3, 4, 5],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=200, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=False, model_type=0, data_type=SINGLE_RANGE_DOPPLER, domain=3,
                     train_index=[1],
                     test_index=[0, 2, 3, 4],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=False, model_type=0, data_type=SINGLE_RANGE_DOPPLER, domain=1,
                     train_index=[0, 1, 2, 3, 4],
                     test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=False, model_type=0, data_type=SINGLE_RANGE_DOPPLER, domain=2,
                     train_index=[0],
                     test_index=[1, 2, 3, 4, 5],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=False, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=3,
                     train_index=[1],
                     test_index=[0, 2, 3, 4],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=False, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=1,
                     train_index=[0, 1, 2, 3, 4],
                     test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=False, model_type=0, data_type=RANGE_ANGLE_IMAGE, domain=2,
                     train_index=[0],
                     test_index=[1, 2, 3, 4, 5],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=False, model_type=3, data_type=COMPLEX_RANGE_DOPPLER, domain=2,
                     train_index=[0],
                     test_index=[1, 2, 3, 4, 5],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)
    new_cross_domain(augmentation=False, model_type=3, data_type=COMPLEX_RANGE_DOPPLER, domain=1,
                     train_index=[0, 1, 2, 3, 4],
                     test_index=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                     # val_index=[5, 6],
                     # test_index=[3, 4, 5, 6, 7, 8, 9],
                     val_time=5,
                     need_test=True,
                     multistream=True, diff=True,
                     attention=True, epoch=100, data_spliter=di_DataSplitter)'''
