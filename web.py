import numpy as np
import pandas as pd
from gensim.models import Word2Vec, FastText
import streamlit as st
import pydeck as pdk
import plotly.graph_objects as go

st.title('Coffee Recommendation System')

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

def make_clickable(url, text):
    return f'<a target="_blank" href="{url}">{text}</a>'

@st.cache(suppress_st_warning=True)
def load_data():
    data = pd.read_csv('coffee_0907_full.csv')
    data = data.dropna(subset=['flavor']).reset_index(drop=True)
    return data

data = load_data()

mode = st.sidebar.selectbox(
    'モード選択',
    ('フレーバー検索', 'インデックス検索')
)

model_type = st.sidebar.selectbox(
    'モデル',
    ('Word2Vec', 'FastText')
)

if model_type == 'Word2Vec':
    model = Word2Vec.load('./word2vec_0909.model')
elif model_type == 'FastText':
    model = pd.read_pickle('./fasttext_flavor_vectors.pkl')

method = st.sidebar.selectbox(
    '重み付けの方法',
    ('なし', 'Countベース', 'TF-IDFベース', 'SCDV', 'SIF')
)

if model_type == 'Word2Vec':
    if method == 'なし':
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
    if method == 'なし':
        flavor_vector_list = pd.read_pickle('./fasttext_normal_vec_list_0907.pkl')
    elif method == 'Countベース':
        flavor_vector_list = pd.read_pickle('./fasttext_count_vec_list_0907.pkl')
    elif method == 'TF-IDFベース':
        flavor_vector_list = pd.read_pickle('./fasttext_tfidf_vec_list_0907.pkl')
    elif method == 'SCDV':
        flavor_vector_list = pd.read_pickle('./fasttext_scdv_vec_list_0907.pkl')
        scdv_word_vectors = pd.read_pickle('./fasttext_scdv_flavor_vectors.pkl')
    elif method == 'SIF':
        flavor_vector_list = pd.read_pickle('./fasttext_sif_vec_list_0907.pkl')

if mode == 'インデックス検索':
    st.header('使い方')
    st.markdown('''
                このモードは、あるコーヒー豆を指定した時に、それと近いものを表示するものです。（主に結果の正しさを見るために使用）
                
                指定したコーヒー豆と同じ国の候補が多いほど、下の地図の赤い丸が大きくなり、出力結果の正しさが直感的に理解できます。
                ''')
    idx = st.sidebar.slider(
        'インデックスの指定',
        0, len(data), 0, step=1
    )

    k = st.sidebar.slider(
        '候補数',
        1, 50, 10
    )

    COLOR_LIST = [
        [50,133,92,200],
        [200,13,13,200]
    ]

    result = get_sim_record(flavor_vector_list, data, idx=idx,top=k)
    result.fillna('null', inplace=True)
    try:
        result['similarity'] = result['similarity'].apply(lambda x : round(x, 3))
        result['per_1g'] = result['per_1g'].apply(lambda x :round(x, 2))
    except:
        pass
    plot_result = result[['similarity', 'country', 'flavor', 'variety', 'process', 'roast', 'per_1g', 'shop']].rename(columns={'similarity':'類似度', 'country':'生産国', 'flavor':'風味', 'variety':'品種', 'process':'精製', 'roast':'焙煎度合い', 'per_1g':'1gあたりの値段', 'shop':'店名'}).reset_index(drop=True)
    show_data = st.radio(
        '出力結果',
        ('表示', '非表示')
    )
    if show_data == '表示':
        st.write(plot_result.to_html(escape=False), unsafe_allow_html=True)
    # st.dataframe(result[['similarity', 'country', 'flavor', 'variety', 'process', 'roast', 'per_1g']].rename(columns={'similarity':'類似度', 'country':'生産国', 'flavor':'風味', 'variety':'品種', 'process':'精製', 'roast':'焙煎度合い', 'per_1g':'1gあたりの値段'}))

    chart = st.radio(
        '表示項目',
        ('地図', '地域割合', '焙煎度合い')
    )
    if chart == '地図':
        sel_country = result.loc[idx, 'country']
        result['color_idx'] = 0
        result.loc[result['country'] == sel_country, 'color_idx'] = 1
        result['color'] = result['color_idx'].map(lambda x : COLOR_LIST[x])

        geo = result[['country', 'lat', 'lon', 'color']]
        geo['cnt'] = geo['country'].map(geo.country.value_counts())
        geo = geo.drop_duplicates(subset='country')

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
    elif chart == '地域割合':
        color_list = ['#4e3518', '#62421e', '#754f24', '#895c2a', '#9c6a30', '#9e7c61']
        area_count = pd.DataFrame(result['area'].value_counts())
        if 'null' in area_count.index:
            area_count.drop(['null'], axis=0, inplace=True)
        fig = go.Figure(data=[go.Pie(labels=area_count.index, values=area_count.area)])
        fig.update_traces(marker=dict(colors=color_list[:len(area_count)]))
        st.plotly_chart(fig, use_container_width=True)
    elif chart == '焙煎度合い':
        color_dict = {
            '浅煎り':'#9c6a30',
            '中浅煎り':'#895c2a',
            '中煎り':'#754f24',
            '中深煎り':'#62421e',
            '深煎り':'#4e3518',
        }
        roast_count = pd.DataFrame(result['roast'].value_counts())
        if 'null' in roast_count.index:
            roast_count.drop(['null'], axis=0, inplace=True)
        if '煎り' in roast_count.index:
            roast_count.drop(['煎り'], axis=0, inplace=True)
        color_list = [color_dict[roast] for roast in roast_count.index]
        fig = go.Figure(data=[go.Pie(labels=roast_count.index, values=roast_count.roast)])
        fig.update_traces(marker=dict(colors=color_list))
        st.plotly_chart(fig, use_container_width=True)

elif mode == 'フレーバー検索':
    st.header('使い方')
    st.markdown('''
                1. 好きなフレーバーを選びます。
                2. そのフレーバーの中で特に好きなものがあれば選択してください。
                3. 選択したフレーバーに近いコーヒー豆が表示されます。
                4. 候補数はサイドメニューのバーをスライドすることで変更できます。
                5. モデルと重み付けの方法を変えることで違う結果を得ることもできます。
                ''')
    if st.checkbox('モデルと重み付けの方法について'):
        st.markdown('''
                    - モデル
                        - Word2Vec
                            - 単語をベクトルで表す際に用いられる一般的な手法。
                        - FastText
                            - Word2Vecよりも細かい分割(単語ではなくサブワード)でベクトル化することができる手法。
                    - 重み付けの方法
                        - なし
                            - 重み付けを行わない単純平均。
                        - Countベース
                            - 文章全体における単語の出現回数が多いものには小さい重みを、少ないものには大きな重みを割り当てる。
                        - TF-IDFベース
                            - ある文章では出現回数が多いが、他の文章ではあまり出現しないような、その文章を特徴付ける単語に大きな重みを割り当てる。
                        - SCDV
                            - 単語のベクトルをクラスタリングすることで潜在トピックに分類し、その情報と他の文章での出現回数を考慮して重みを割り当てる。
                        - SIF
                            - モデルを学習する際に使用した文章における単語の出現回数に基づいて重みを割り当てる。(性能が良いことが知られている)
                    ''')
        
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
        if model_type == 'Word2Vec':
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
        
        elif model_type == 'FastText':
            for flavor in flavors:
                for flavor_name in flavor_dict[flavor]:
                    if flavor_name in model.keys():
                        if flavor_name in detail_flavor:
                            if method == 'SCDV':
                                flavor_vector += 3 * scdv_word_vectors[flavor_name]
                            else:
                                flavor_vector += 3 * model[flavor_name]
                        else:
                            if method == 'SCDV':
                                flavor_vector += scdv_word_vectors[flavor_name]
                            else:
                                flavor_vector += model[flavor_name]
                        word_cnt += 1
            flavor_vector /= word_cnt
    
    k = st.sidebar.slider(
        '候補数',
        1, 50, 10
    )

    result = get_flavor_record(flavor_vector, flavor_vector_list, data ,top=k)
    result.fillna('null', inplace=True)
    try:
        result['similarity'] = result['similarity'].apply(lambda x : round(x, 3))
        result['per_1g'] = result['per_1g'].apply(lambda x :round(x, 2))
    except:
        pass
    plot_result = result[['similarity', 'country', 'flavor', 'variety', 'process', 'roast', 'per_1g', 'shop', 'url']].rename(columns={'similarity':'類似度', 'country':'生産国', 'flavor':'風味', 'variety':'品種', 'process':'精製', 'roast':'焙煎度合い', 'per_1g':'1gあたりの値段', 'shop':'店名', 'url':'URL'}).reset_index(drop=True)
    plot_result['URL'] = plot_result['URL'].apply(lambda x : make_clickable(x, 'click here'))
    show_data = st.radio(
        '出力結果',
        ('表示', '非表示')
    )
    if show_data == '表示':
        st.write(plot_result.to_html(escape=False), unsafe_allow_html=True)
    # st.dataframe(result[['similarity', 'country', 'flavor', 'variety', 'process', 'roast', 'per_1g', 'url']].rename(columns={'similarity':'類似度', 'country':'生産国', 'flavor':'風味', 'variety':'品種', 'process':'精製', 'roast':'焙煎度合い', 'per_1g':'1gあたりの値段', 'url':'URL'}))

    chart = st.radio(
        '表示項目',
        ('地図', '地域割合', '焙煎度合い')
    )
    if chart == '地図':
        geo = result[['country', 'lat', 'lon']]
        geo['cnt'] = geo['country'].map(geo.country.value_counts())
        geo = geo.drop_duplicates(subset='country')

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
                    get_fill_color=[50,133,92,200],
                    get_line_color=[0, 0, 0],
                    get_radius='cnt',
                    radius_scale=100000,
                )
            ],
            tooltip={"text":"{country}:{cnt}"}
        ))
    
    elif chart == '地域割合':
        color_list = ['#4e3518', '#62421e', '#754f24', '#895c2a', '#9c6a30', '#9e7c61']
        area_count = pd.DataFrame(result['area'].value_counts())
        if 'null' in area_count.index:
            area_count.drop(['null'], axis=0, inplace=True)
        fig = go.Figure(data=[go.Pie(labels=area_count.index, values=area_count.area)])
        fig.update_traces(marker=dict(colors=color_list[:len(area_count)]))
        st.plotly_chart(fig, use_container_width=True)
    elif chart == '焙煎度合い':
        color_dict = {
            '浅煎り':'#9c6a30',
            '中浅煎り':'#895c2a',
            '中煎り':'#754f24',
            '中深煎り':'#62421e',
            '深煎り':'#4e3518',
        }
        roast_count = pd.DataFrame(result['roast'].value_counts())
        if 'null' in roast_count.index:
            roast_count.drop(['null'], axis=0, inplace=True)
        if '煎り' in roast_count.index:
            roast_count.drop(['煎り'], axis=0, inplace=True)
        color_list = [color_dict[roast] for roast in roast_count.index]
        fig = go.Figure(data=[go.Pie(labels=roast_count.index, values=roast_count.roast)])
        fig.update_traces(marker=dict(colors=color_list))
        st.plotly_chart(fig, use_container_width=True)
# if st.checkbox('開発ログを表示'):
#     st.write('10/01 インデックス検索とフレーバー検索、重み付けの方法を追加')
#     st.write('10/02 味の詳細検索を追加')
#     st.write('10/23 SCDVを実装')
#     st.write('11/5 Fasttextの追加')