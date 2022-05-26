import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Input, Model, Sequential
import numpy as np
from sklearn.preprocessing import LabelEncoder


# from sklearn.model_selection import train_test_split
# %%
def cx_pref_model():
    num_of_age = 86
    age = Input(shape=1, name="age")  # Variable-length sequence of ints
    other_customer_input = Input(shape=7, name="customer_info")
    customer_clusters = Input(shape=4, name="customer_clusters")

    cluster_norm = layers.BatchNormalization()(customer_clusters)
    other_cx = layers.Dense(7, kernel_regularizer='l1', use_bias=False,
                            kernel_initializer=tf.keras.initializers.Zeros())(other_customer_input)
    cx_clust = layers.Dense(15, kernel_regularizer='l1')(cluster_norm)

    # Embed each word in the text into a 64-dimensional vector
    age_embedding = layers.Embedding(num_of_age, 16)(age)

    # f_1 = layers.Flatten()(customer_embedding)
    f_2 = layers.Flatten()(age_embedding)
    # Merge all available features into a single large vector via concatenation
    concat = layers.concatenate(inputs=[f_2, other_cx, cx_clust])
    x = layers.Dense(450, kernel_regularizer='l1')(concat)
    # Stick a pref classifier on top of the features
    pref = layers.Dense(620, name="pref", kernel_regularizer='l1', kernel_initializer=tf.keras.initializers.Zeros())(x)

    # Instantiate an end-to-end model predicting both priority and department
    model = tf.keras.Model(
        inputs=[age, other_customer_input, customer_clusters],
        outputs=[pref]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(.9, .5, amsgrad=True),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=tf.keras.metrics.Accuracy(), )
    return model


def train_cx_pref_model(pref_model):
    data = pd.merge(pd.read_csv('../inputs/customer_info.csv'), pd.read_csv('../inputs/pref.csv'), on='customer_id')

    x = {"age": data['age'], "customer_info": data[
        ['FN', 'fashion_news_frequency_Regularly', 'club_member_status_LEFT CLUB', 'club_member_status_PRE-CREATE',
         'female', 'male', 'children']],
         "customer_clusters": data[['full_order', 'mens_order', 'womens_order', 'fam_order']]}
    Y = {"pref": data.drop(
        columns=['customer_id', 'FN', 'age', 'fashion_news_frequency_Regularly', 'club_member_status_LEFT CLUB',
                 'club_member_status_PRE-CREATE', 'female', 'male', 'children', 'full_order', 'mens_order',
                 'womens_order', 'fam_order'])}
    print(pref_model.summary())
    pref_model.fit(x, Y,
                   epochs=15,
                   batch_size=100000)
    return pref_model, data


def pref_article_model(article_file_to_load, save_model_as_filename):
    """
    takes the preference df of 1 hot encoded variables and matches it to the respective articles
    :param article_file_to_load: csv file name eg: '../inputs/clothing_df'
    :param save_model_as_filename: the path to the saved model
    :return: model
    """

    data = pd.read_csv(article_file_to_load)
    # data, cols_x, cols_y = build_1_hot_article_array(data_x)
    le = LabelEncoder()
    le.fit(data['article_id'])
    data['labels'] = le.transform(data['article_id'])
    pref = Input(shape=620, name="pref")
    # x = layers.Dense(5)(pref)
    art = layers.Dense(len(data.article_id.unique()), name="articles")(pref)

    # Instantiate an end-to-end model predicting both priority and department
    model = tf.keras.Model(
        inputs=[pref],
        outputs=[art]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(2, .8, amsgrad=True),
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.AUC()])
    x = {"pref": data.drop(columns=['article_id', 'labels'])}
    y = {"articles": data['labels']}
    print(model.summary())
    model.fit(x, y, epochs=30, batch_size=50000)
    model.save(save_model_as_filename)
    return model, data


def combine_sequentially(models_to_load_combine={"../inputs/cx_pref": True, "../inputs/pref_art": False},
                         file_path='/'):
    """
    combines pretrained and saved models :param models_to_load_combine: Takes a dic of {model file names:Trainable}
    loads models front to back and combines layers :param file_path: file path to :return: combined model un-compiled
    """
    retrained_model = Sequential()

    for model_names in models_to_load_combine.keys():
        model = tf.keras.models.load_model(file_path + model_names)
        for layer in model.layers[:-1]:
            layer.trainable = models_to_load_combine[model_names]
            retrained_model.add(layer)
    retrained_model.compile(optimizer='adam',
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                            metrics=['accuracy'])
    retrained_model.summary()
    return retrained_model


def train_gender(pref_model, epoch, m_f='m'):
    if m_f == 'm':
        data = pd.read_csv('../inputs/male_cx_data.csv')
        gender = 'male'
    elif m_f == 'f':
        data = pd.read_csv('../inputs/female_cx_data.csv')
        gender = 'female'
    else:
        raise ValueError('try again')

    x = {"age": data['age'], "customer_info": data[
        ['FN', 'fashion_news_frequency_Regularly', 'club_member_status_LEFT CLUB', 'club_member_status_PRE-CREATE',
         'female', 'male', 'children']]}
    Y = {"pref": data.drop(
        columns=['customer_id', 'FN', 'age', 'fashion_news_frequency_Regularly', 'club_member_status_LEFT CLUB',
                 'club_member_status_PRE-CREATE', 'female', 'male', 'children'])}

    pref_model.fit(x, Y,
                   epochs=epoch,
                   batch_size=1)
    pref_model.save(f'{gender}_model')
    return pref_model


def test_model(data, file_path_to_input_model='cx_pref', ):
    x = {"age": data['age'], "customer_info": data[
        ['FN', 'fashion_news_frequency_Regularly', 'club_member_status_LEFT CLUB', 'club_member_status_PRE-CREATE',
         'female', 'male', 'children']],
         "customer_clusters": data[['full_order', 'mens_order', 'womens_order', 'fam_order']]}
    y = pd.read_csv('../inputs/test_df.csv')
    tf.keras.model.load_model(file_path_to_input_model)
    model = combine_sequentially(models_to_load_combine={file_path_to_input_model: False, 'pref_art': False})
    results = model.evaluate(x, y, batch_size=1)
    print(results)


def make_final_predictions(file_path_to_input_model='cx_pref',
                           model_input_df_path='../inputs/female_cx_data.csv'):
    tf.keras.model.load_model(file_path_to_input_model)
    model = combine_sequentially(models_to_load_combine={file_path_to_input_model: False, 'final_pref_art': False})
    predictions = model.predict
    return model


def main():
    # general preferences
    cx_pref, input_data = train_cx_pref_model(cx_pref_model())
    cx_pref.save('cx_pref')

    # train general article choices
    pref_art = pref_article_model('../inputs/test_arts.csv', '2018_target')

    # gender-specific training
    # cx_pref = tf.keras.models.load_model('cx_pref')
    # male_pref = train_gender(pref_model=cx_pref, epoch=50, m_f='m')
    # female_pref = train_gender(pref_model=cx_pref, epoch=50, m_f='f')

    # test on full timeframe dataset
    # test_model(file_path_to_input_model='male', model_input_df_path='../inputs/male_cx_data.csv')
    # test_model(file_path_to_input_model='female', model_input_df_path='../inputs/female_cx_data.csv')
    test_model(file_path_to_input_model='../Models/cx_pref/saved_model.pb', data=input_data[
        ['age','FN', 'fashion_news_frequency_Regularly', 'club_member_status_LEFT CLUB', 'club_member_status_PRE-CREATE',
         'female', 'male', 'children', 'full_order', 'mens_order', 'womens_order', 'fam_order']])

    # rebuild for slimmed down
    final = make_final_predictions()
    # predict on slimmed down articles


if __name__ == '__main__':
    main()
