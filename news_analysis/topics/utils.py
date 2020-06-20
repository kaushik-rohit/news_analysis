import pandas as pd
import os
import ast


def get_category(topic, mapp):
    category = mapp.loc[mapp['Topic'] == topic]
    return category['Category'].iloc[0]


def preprocess_speech_data(path, mapp):
    df = pd.read_csv(path, index_col=0)
    df = df.drop(['speaker'], axis=1)
    df['topic'] = df.apply(lambda x: x['topic'].strip(), axis=1)
    df['topic'] = df.apply(lambda x: get_category(x['topic'], mapp), axis=1)
    df.to_csv(path, index=False)


def sample_balanced_dataset(path):
    """
    Randomly samples speeches such that each topic has equal representation in the sample

    :param path: path of csv file containing speech data

    :return:
    pandas Dataframe: a random sampled records of input csv
    """
    df = pd.read_csv(os.path.join(path, '2012_speech.csv'), index_col=0)


def calculate_accuracy(preds):
    topics = preds.topic.unique()
    topics_count = preds.topic.value_counts()

    preds['predicted_topic'] = preds['predicted_topic'].apply(lambda x: ast.literal_eval(x))
    true_topic = preds['topic'].values
    pred_topic = preds['predicted_topic'].values
    total = len(true_topic)
    correct_pred_top1 = 0
    correct_pred_top3 = 0

    accuracy_by_topic = {topic: [0, 0] for topic in topics}

    for i in range(total):
        if true_topic[i] == pred_topic[i][0][0]:
            correct_pred_top1 += 1
            correct_pred_top3 += 1
            accuracy_by_topic[true_topic[i]][0] += 1
            accuracy_by_topic[true_topic[i]][1] += 1
        elif true_topic[i] == pred_topic[i][1][0]:
            correct_pred_top3 += 1
            accuracy_by_topic[true_topic[i]][1] += 1
        elif true_topic[i] == pred_topic[i][2][0]:
            correct_pred_top3 += 1
            accuracy_by_topic[true_topic[i]][1] += 1

    rows = []
    for key, value in accuracy_by_topic.items():
        rows += [[key, value[0]/topics_count[key], value[1]/topics_count[key]]]

    df = pd.DataFrame(rows, columns=['topic', 'top1-accuracy', 'top3-accuracy'])
    print(df)
    print('top1-accuracy: {}, top3-accuracy: {}'.format(correct_pred_top1/total, correct_pred_top3/total))
