import numpy as np
import pandas as pd
from gensim.models import Word2Vec, FastText
import streamlit as st
import pydeck as pdk

st.title('Coffee Recommendation System Ver0.3')

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_sim_record(flavor_vector_list, data, idx, top=5):
    sim_list = []
    for i in range(len(flavor_vector_list)):
        if idx == i:
            continue
        sim = cos_sim(flavor_vector_list[idx], flavor_vector_list[i])
        sim_list.append((i, sim))
    sim_list = sorted(sim_list, key=lambda x: -np.inf if np.isnan(x[1]) else x[1], reverse=True)
    sim_idx = [idx] + [x[0] for x in sim_list[:top]]
    result = data.iloc[sim_idx, :]
    result['similarity'] = [1] + [x[1] for x in sim_list[:top]]
    return result

def get_flavor_record(flavor_vector, flavor_vector_list, data, top=5):
    sim_list = []
    for i in range(len(flavor_vector_list)):
        sim = cos_sim(flavor_vector, flavor_vector_list[i])
        sim_list.append((i, sim))
    sim_list = sorted(sim_list, key=lambda x: -np.inf if np.isnan(x[1]) else x[1], reverse=True)
    sim_idx = [x[0] for x in sim_list[:top]]
    result = data.iloc[sim_idx, :]
    result['similarity'] = [x[1] for x in sim_list[:top]]
    return result

@st.cache(suppress_st_warning=True)
def load_data():
    data = pd.read_csv('coffee_0907_full.csv')
    data = data.dropna(subset=['flavor']).reset_index(drop=True)
    return data

data = load_data()

mode = st.sidebar.selectbox(
    'モード選択',
    ('インデックス検索', 'フレーバー検索')
)

model_type = st.sidebar.selectbox(
    'モデル',
    ('Word2Vec', 'FastText')
)

if model_type == 'Word2Vec':
    model = Word2Vec.load('word2vec_0909.model')
elif model_type == 'FastText':
    model = FastText.load("fasttext_1023.model")

method = st.sidebar.selectbox(
    '重み付けの方法',
    ('通常', 'Countベース', 'TF-IDFベース', 'SCDV', 'SIF')
)

if model_type == 'Word2Vec':
    if method == '通常':
        flavor_vector_list = pd.read_pickle('./normal_vec_list_0907.pkl')
    elif method == 'Countベース':
        flavor_vector_list = pd.read_pickle('./count_vec_list_0907.pkl')
    elif method == 'TF-IDFベース':
        flavor_vector_list = pd.read_pickle('./tfidf_vec_list_0907.pkl')
    elif method == 'SCDV':
        flavor_vector_list = pd.read_pickle('./scdv_vec_list_0907.pkl')
        scdv_word_vectors = pd.read_pickle('./scdv_word_vectors_0907.pkl')
    elif method == 'SIF':
        flavor_vector_list = pd.read_pickle('./sif_vec_list_0907.pkl')

if model_type == 'FastText':
    if method == '通常':
        flavor_vector_list = pd.read_pickle('./fasttext_normal_vec_list_0907.pkl')
    elif method == 'Countベース':
        flavor_vector_list = pd.read_pickle('./fasttext_count_vec_list_0907.pkl')
    elif method == 'TF-IDFベース':
        flavor_vector_list = pd.read_pickle('./fasttext_tfidf_vec_list_0907.pkl')
    elif method == 'SCDV':
        flavor_vector_list = pd.read_pickle('./fasttext_scdv_vec_list_0907.pkl')
        scdv_word_vectors = pd.read_pickle('./fasttext_scdv_word_vectors_0907.pkl')
    elif method == 'SIF':
        flavor_vector_list = pd.read_pickle('./fasttext_sif_vec_list_0907.pkl')

if mode == 'インデックス検索':
    idx = st.sidebar.slider(
        'インデックスの指定',
        0, len(data), 0, step=1
    )

    k = st.sidebar.slider(
        '候補数',
        1, 50, 10
    )

    COLOR_LIST = [
        [29,103,87,200],
        [200,13,13,200]
    ]

    result = get_sim_record(flavor_vector_list, data, idx=idx,top=k)
    result.fillna('null', inplace=True)
    st.dataframe(result[['similarity', 'country', 'flavor', 'per_1g', 'roast']])

    sel_country = result.loc[idx, 'country']
    result['color_idx'] = 0
    result.loc[result['country'] == sel_country, 'color_idx'] = 1
    result['color'] = result['color_idx'].map(lambda x : COLOR_LIST[x])

    geo = result[['country', 'lat', 'lon', 'color']]
    geo['cnt'] = geo['country'].map(geo.country.value_counts())
    # geo['cnt_sqrt'] = np.sqrt(geo['cnt'])
    geo = geo.drop_duplicates(subset='country')

    # 地図
    # st.map(result[['lat', 'lon']])

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/hiromu/ckfqgnutx0rbm19o703uqrdjf',
        initial_view_state=pdk.ViewState(
            latitude=0,
            longitude=0,
            zoom=1
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=geo,
                get_position=['lon', 'lat'],
                pickable=True,
                opacity=0.8,
                stroked=True,
                filled=True,
                get_fill_color='color',
                get_line_color=[0, 0, 0],
                get_radius='cnt',
                radius_scale=100000,
            )
        ],
        tooltip={"text":"{country}:{cnt}"}
    ))

elif mode == 'フレーバー検索':
    flavor_dict = {
        'フローラル':['カモミール', 'ローズ', 'ジャスミン', 'フローラル', '花'],
        'フルーティー':['ブラックベリー', 'ラズベリー', 'ブルーベリー', 'ストロベリー', 'ベリー', 'レーズン', 'プルーン', 'ドライフルーツ', 'ココナッツ', 'チェリー', 'ザクロ', 'パイナップル', 'グレープ', 'マスカット', 'リンゴ', 'りんご', 'アップル', '桃', '洋梨', 'グレープフルーツ', 'オレンジ', 'レモン', 'ライム', 'プラム', '柑橘'],
        'スイート':['甘み', '甘味', 'バニラ', 'ハチミツ', '蜂蜜', 'キャラメル', 'メープルシロップ', 'シロップ', '黒糖', '糖蜜', 'カラメル', '蜜'],
        'ナッツ＆チョコ':['チョコ', 'チョコレート', 'ミルクチョコ', 'ミルクチョコレート', 'ココア', 'カカオ', 'ダーク', 'ナッツ', 'アーモンド', 'ヘーゼルナッツ', 'ピーナッツ'],
        'スパイシー':['クローブ', 'シナモン', 'ナツメグ', 'アニス', 'スパイス', 'スパイシー']
    }
    flavors = st.multiselect(
        '好きなフレーバー', ['フローラル', 'フルーティー', 'スイート', 'ナッツ＆チョコ', 'スパイシー'], ['フローラル']
    )
    detail_flavor = []
    if 'フローラル' in flavors:
        froral_flavor = st.multiselect(
            '詳細選択（フローラル）', ['カモミール', 'ローズ', 'ジャスミン']
        )
        detail_flavor.extend(froral_flavor)
    if 'フルーティー' in flavors:
        fruity_flavor = st.multiselect(
            '詳細選択（フルーティー）', ['ブラックベリー', 'ラズベリー', 'ブルーベリー', 'ストロベリー', 'レーズン', 'プルーン', 'ココナッツ', 'チェリー', 'ザクロ', 'パイナップル', 'グレープ', 'マスカット', 'リンゴ', '桃', '洋梨', 'グレープフルーツ', 'オレンジ', 'レモン', 'ライム', 'プラム', '柑橘']
        )
        detail_flavor.extend(fruity_flavor)
    if 'スイート' in flavors:
        sweet_flavor = st.multiselect(
            '詳細選択（スイート）', ['バニラ', 'ハチミツ', 'キャラメル', 'メープルシロップ', '黒糖']
        )
        detail_flavor.extend(sweet_flavor)
    if 'ナッツ＆チョコ' in flavors:
        nut_cocoa_flavor = st.multiselect(
            '詳細選択（ナッツ＆チョコ）', ['チョコレート', 'ミルクチョコレート', 'ココア', 'カカオ', 'ナッツ', 'アーモンド', 'ヘーゼルナッツ', 'ピーナッツ']
        )
        detail_flavor.extend(nut_cocoa_flavor)
    if 'スパイシー' in flavors:
        spicy_flavor = st.multiselect(
            '詳細選択（スパイシー）', ['クローブ', 'シナモン', 'ナツメグ', 'アニス']
        )
        detail_flavor.extend(spicy_flavor)
    st.write('*詳細選択は一部反映されないものも含まれています')
    
    flavor_vector = np.zeros(len(flavor_vector_list[0]))
    word_cnt = 0
    if flavors != []:
        for flavor in flavors:
            for flavor_name in flavor_dict[flavor]:
                if flavor_name in model.wv.vocab.keys():
                    if flavor_name in detail_flavor:
                        if method == 'SCDV':
                            flavor_vector += 3 * scdv_word_vectors[flavor_name]
                        else:
                            flavor_vector += 3 * model.wv[flavor_name]
                    else:
                        if method == 'SCDV':
                            flavor_vector += scdv_word_vectors[flavor_name]
                        else:
                            flavor_vector += model.wv[flavor_name]
                    word_cnt += 1
        flavor_vector /= word_cnt
    
    k = st.sidebar.slider(
        '候補数',
        1, 50, 10
    )

    result = get_flavor_record(flavor_vector, flavor_vector_list, data ,top=k)
    result.fillna('null', inplace=True)
    st.dataframe(result[['similarity', 'country', 'flavor', 'per_1g', 'roast', 'url']])

    geo = result[['country', 'lat', 'lon']]
    geo['cnt'] = geo['country'].map(geo.country.value_counts())
    # geo['cnt_sqrt'] = np.sqrt(geo['cnt'])
    geo = geo.drop_duplicates(subset='country')

    # 地図
    # st.map(result[['lat', 'lon']])

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/hiromu/ckfqgnutx0rbm19o703uqrdjf',
        initial_view_state=pdk.ViewState(
            latitude=0,
            longitude=0,
            zoom=1
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=geo,
                get_position=['lon', 'lat'],
                pickable=True,
                opacity=0.8,
                stroked=True,
                filled=True,
                get_fill_color=[29,103,87,200],
                get_line_color=[0, 0, 0],
                get_radius='cnt',
                radius_scale=100000,
            )
        ],
        tooltip={"text":"{country}:{cnt}"}
    ))
    
if st.checkbox('開発ログを表示'):
    st.write('10/01 インデックス検索とフレーバー検索、重み付けの方法を追加')
    st.write('10/02 味の詳細検索を追加')
    st.write('10/23 SCDVを実装')
    st.write('11/5 Fasttextの追加')