import numpy as np
import pandas as pd


def preprocessing():
    def customer_data():
        cx_data = pd.read_csv('../h-and-m-personalized-fashion-recommendations/customers.csv')
        # club_member_status = ['ACTIVE' nan 'PRE-CREATE' 'LEFT CLUB']
        # fashion_news_frequency = ['NONE' 'Regularly' nan 'Monthly' 'None'] -> ['Regularly','NONE']
        cx_data = cx_data.drop(columns=['Active', 'postal_code'])

        # Newsletter to categorical
        cx_data['fashion_news_frequency'] = cx_data['fashion_news_frequency'].str.replace(pat='None', repl='NONE',
                                                                                          case=True)
        cx_data['fashion_news_frequency'] = cx_data['fashion_news_frequency'].str.replace(pat='Monthly',
                                                                                          repl='Regularly')
        cx_data['fashion_news_frequency'] = cx_data['fashion_news_frequency'].fillna(value='NONE')
        cx_data = pd.get_dummies(cx_data, columns=['fashion_news_frequency', 'club_member_status'], drop_first=True)

        # Finds relative frequency of the
        art = pd.read_csv('../h-and-m-personalized-fashion-recommendations/articles.csv',
                          usecols=['index_group_no', 'section_no', 'article_id'])
        art['female'] = (
                ((art['index_group_no'] == 1) | (art['index_group_no'] == 2)) | (art['section_no'] == 5)).astype(
            int)

        art['male'] = ((art['index_group_no'] == 3) | (art['section_no'] == 22)).astype(int)
        art['children'] = ((art['index_group_no'] == 4) | (art['section_no'] == 29)).astype(int)
        df = pd.merge(pd.read_csv('../h-and-m-personalized-fashion-recommendations/transactions_train.csv'),
                      art, on='article_id', how='left')

        cx_data = pd.merge(cx_data, df[['customer_id', 'female', 'male', 'children']].groupby(by='customer_id',
                                                                                              as_index=True).sum(),
                           on='customer_id', how='left')
        cx_data[['female', 'male', 'children']] = cx_data[['female', 'male', 'children']].div(
            cx_data[['female', 'male', 'children']].sum(axis=1), axis=0)
        cx_data['age'] = cx_data['age'] - 16
        cx_data = cx_data.fillna(0)
        return cx_data

    def articles():
        clothing_df = pd.read_csv('../h-and-m-personalized-fashion-recommendations/articles.csv',
                                  usecols=['article_id', 'product_code', 'product_group_name', 'product_type_no',
                                           'graphical_appearance_no', 'colour_group_code',
                                           'perceived_colour_master_id',
                                           'department_no', 'section_no', 'garment_group_no'])

        to_combine = ['Garment and Shoe care', 'Furniture', 'Stationery', 'Interior textile', 'Fun']
        for items in to_combine:
            clothing_df['product_group_name'].str.replace(items, 'Items')
        clothing_df = pd.get_dummies(clothing_df, columns=['product_type_no', 'graphical_appearance_no',
                                                           'colour_group_code', 'product_group_name',
                                                           'perceived_colour_master_id',
                                                           'department_no', 'section_no', 'garment_group_no'],
                                     drop_first=True)
        clothing_df.to_csv('../inputs/clothing.csv', index=False)
        return clothing_df

    def transactions():
        """
        t_dat,customer_id,article_id,price,sales_channel_id
        :return:
        """

        def find_max_min_date(df, by):
            df['date'] = pd.to_datetime(df['t_dat'], format="%Y-%m-%d")
            gb_object = (df.groupby(by, as_index=False)
                         .agg(**{'First_Date': ('date', 'first'),
                                 'Last_Date': ('date', 'last')}))
            gb_object['duration'] = (gb_object['Last_Date'] - gb_object['First_Date']).dt.days
            return gb_object

        trans = pd.read_csv('../h-and-m-personalized-fashion-recommendations/transactions_train.csv')
        first_last_dates = find_max_min_date(trans, 'article_id')
        trans = trans.drop(columns=['t_dat', 'price'])
        trans = pd.get_dummies(trans, columns=['sales_channel_id'], drop_first=True)
        first_last_dates.to_csv('first_last_date.csv', index=False)
        return trans, first_last_dates

    def make_and_write_input_output():
        cx_df = customer_data()
        cx_factors = pd.DataFrame()
        cx_factors['codes'], cx_factors['index'] = pd.factorize(cx_df.customer_id)
        cx_df['customer_id'] = cx_df['customer_id'].map(cx_factors.set_index('index')['codes'])
        cx_df.to_csv('customer_info.csv', index=False)
        cx_factors.to_csv('customer_factors.csv', index=False)
        trans, first_last_dates = transactions()
        trans['customer_id'] = trans['customer_id'].map(cx_factors.set_index('index')['codes'])

        df = pd.merge(trans, cx_df,
                      how='left', on='customer_id')

        df = pd.merge(df, articles(), how='left', on='article_id')
        df.to_csv('preferences.csv', index=False)

    make_and_write_input_output()


def spaghetti():
    x = pd.read_csv('pref.csv')
    cx_id = x['customer_id']
    x = x.drop(columns='customer_id').div(x.sum(axis=1), axis=0)
    print()
    x['customer_id'] = cx_id
    x.to_csv('pref.csv', index=False)


def split_by_gender():
    def articles(section_subset):
        clothing_df = pd.read_csv('../h-and-m-personalized-fashion-recommendations/articles.csv',
                                  usecols=['article_id', 'product_code', 'product_group_name', 'product_type_no',
                                           'graphical_appearance_no', 'colour_group_code',
                                           'perceived_colour_master_id',
                                           'department_no', 'section_no', 'garment_group_no'])

        to_combine = ['Garment and Shoe care', 'Furniture', 'Stationery', 'Interior textile', 'Fun']
        for items in to_combine:
            clothing_df['product_group_name'].str.replace(items, 'Items')
        clothing_df = clothing_df[~clothing_df.section_no.isin(section_subset)]
        clothing_df = pd.get_dummies(clothing_df, columns=['product_type_no', 'graphical_appearance_no',
                                                           'colour_group_code', 'product_group_name',
                                                           'perceived_colour_master_id',
                                                           'department_no', 'section_no', 'garment_group_no'],
                                     drop_first=True)
        return clothing_df

    # Split between gender and family
    cx = pd.read_csv('../inputs/customer_info.csv')
    male = cx[cx['male'] > .75]
    cx = cx[cx['male'] < .75]

    female = cx[cx['female'] > .75]
    cx = cx[cx['female'] < .75]
    pref = pd.read_csv('../inputs/pref.csv')

    male = pd.merge(male, pref, how='left', on='customer_id')
    male.to_csv('../inputs/male_cx_data.csv', index=False)

    female = pd.merge(female, pref, how='left', on='customer_id')
    female.to_csv('../inputs/female_cx_data.csv', index=False)

    cx = pd.merge(cx, pref, how='left', on='customer_id')
    cx.to_csv('../inputs/family_cx_info.csv', index=False)

    male_sections = [26, 22, 31, 55, 21, 25, 23, 27, 20, 56, 29, 30, 24, 28]
    male = articles(male_sections)
    female_sections = [16, 61, 62, 8, 66, 51, 65, 52, 60, 58, 15, 53, 57, 18, 64, 11, 50, 19, 6, 80, 14, 82, 97, 70, 71,
                       4, 17]
    female = articles(female_sections)

    male.to_csv('../inputs/male_clothing.csv', index=False)
    female.to_csv('../inputs/female_clothing.csv', index=False)
    female = pd.get_dummies(female['article_id'])
    male = pd.get_dummies(male['article_id'])
    female.to_csv('../inputs/female_target.csv', index=False)
    male.to_csv('../inputs/male_target.csv', index=False)
    cx = pd.get_dummies(cx['article_id'])
    cx.to_csv('../inputs/art_target.csv', index=False)


def dropping_mothballed_articles():
    dates = pd.read_csv('../inputs/first_last_date.csv')
    dates['Last_Date'] = pd.to_datetime(dates['Last_Date'], exact=False)
    dates = dates[dates['Last_Date'] >= '2020-08-27']
    print('got dates')

    male = pd.read_csv('../inputs/male_target.csv')
    male = pd.merge(male, dates, on='article_id', how='inner')
    male.to_csv('../inputs/male_final_subset.csv', index=False)

    female = pd.read_csv('../inputs/female_target.csv')
    female = pd.merge(female, dates, on='article_id', how='inner')
    female.to_csv('../inputs/female_final_subset.csv', index=False)

    dates.drop(columns=['First_Date', 'Last_Date', 'duration'], inplace=True)

    out_df = pd.get_dummies(dates['article_id'])
    print('writing art')
    out_df.to_csv('../inputs/cx_id_final_subset.csv', index=False)


def make_testing_df():
    import polars as ps
    tras = pd.read_csv('../h-and-m-personalized-fashion-recommendations/transactions_train.csv', usecols=['article_id', 'customer_id'])
    cx_factors = pd.read_csv('../inputs/customer_factors.csv')
    tras['customer_id'] = tras['customer_id'].map(cx_factors.set_index('index')['codes'])
    print('final_df')
    art=(pd.read_csv('../inputs/art_target.csv')).transpose()
    transactions = pd.merge(tras,art, left_on='article_id',right_index=True)
    transactions = transactions.groupby('customer_id').sum()
    transactions = transactions.drop('article_id')
    transactions = transactions.merge(transactions,ps.read_csv('../inputs/customer_info.csv'), on='customer_id')
    transactions.to_csv('../inputs/test_df.csv')
    return transactions


if __name__ == '__main__':
    preprocessing()
    # input('go run r')
    # spaghetti()
    # split_by_gender()
    # dropping_mothballed_articles()
    make_testing_df()
