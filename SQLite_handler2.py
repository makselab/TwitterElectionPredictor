# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    SQLite_handler2.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Zhenkun <zhenkun91@outlook.com>            +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2019/06/07 20:40:05 by Kay Zhou          #+#    #+#              #
#    Updated: 2021/10/14 19:42:05 by Zhenkun          ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# Prediction with updated queries.


import os
import unicodedata
from pathlib import Path
from pprint import pprint
from shutil import copyfile

import joblib
import pendulum
from bs4 import BeautifulSoup
from file_read_backwards import FileReadBackwards
from sqlalchemy import (Column, DateTime, Float, Integer, String, Text, and_,
                        create_engine, desc, exists, or_, text)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.exc import NoResultFound
from tqdm import tqdm

from my_weapon import *

try:
    import ujson as json
except:
    print("No ujson")


Base = declarative_base()

official_twitter_clients = set([
    'Twitter for iPhone',
    'Twitter for Android',
    'Twitter Web Client',
    'Twitter Web App',
    'Twitter for iPad',
    'Mobile Web (M5)',
    'TweetDeck',
    'Mobile Web',
    'Mobile Web (M2)',
    'Twitter for Windows',
    'Twitter for Windows Phone',
    'Twitter for BlackBerry',
    'Twitter for Android Tablets',
    'Twitter for Mac',
    'Twitter for BlackBerry®',
    'Twitter Dashboard for iPhone',
    'Twitter for iPhone',
    'Twitter Ads',
    'Twitter for  Android',
    'Twitter for Apple Watch',
    'Twitter Business Experience',
    'Twitter for Google TV',
    'Chirp (Twitter Chrome extension)',
    'Twitter for Samsung Tablets',
    'Twitter for MediaTek Phones',
    'Google',
    'Facebook',
    'Twitter for Mac',
    'iOS',
    'Instagram',
    'Vine - Make a Scene',
    'Tumblr',
])


class Tweet(Base):
    __tablename__ = "tweets"
    tweet_id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    dt = Column(DateTime)
    proK = Column(Float)
    proM = Column(Float)
    hashtags = Column(String)
    source = Column(String(50))


class Tweet0208(Base):
    __tablename__ = "tweets0208"
    tweet_id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    dt = Column(DateTime)
    proK = Column(Float)
    proM = Column(Float)
    hashtags = Column(String)
    source = Column(String(50))
    

class Tweet0808(Base):
    __tablename__ = "tweets0808"
    tweet_id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    dt = Column(DateTime)
    proK = Column(Float)
    proM = Column(Float)
    hashtags = Column(String)
    source = Column(String(50))


class Tweet3(Base):
    __tablename__ = "tweets3"
    tweet_id = Column(Integer, primary_key=True)
    dt = Column(DateTime)
    proA3 = Column(Float)


class Retweet(Base):
    __tablename__ = "retweets"
    tweet_id = Column(Integer, primary_key=True)
    dt = Column(DateTime)
    user_id = Column(Integer)
    ori_tweet_id = Column(Integer)
    ori_user_id = Column(Integer)


class Source(Base):
    __tablename__ = "sources"
    tweet_id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    source = Column(String(50))


class Term(Base):
    __tablename__ = "terms"
    name = Column(String(100), primary_key=True)
    proK = Column(Integer)
    proM = Column(Integer)
    unclassified = Column(Integer)


class Month_Term(Base):
    __tablename__ = "month_terms"
    name = Column(String(100), primary_key=True)
    proK = Column(Integer)
    proM = Column(Integer)
    unclassified = Column(Integer)


class New_clas(Base):
    __tablename__ = "New_clas2_20190415"
    tweet_id = Column(Integer, primary_key=True)
    user_id = Column(Integer)
    dt = Column(DateTime)
    proK = Column(Float)
    proM = Column(Float)
    # proK3 = Column(Float)
    # proM3 = Column(Float)
    # proA3 = Column(Float)
    # hashtags = Column(String)
    # source = Column(String(50))


class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True)
    tweet_id = Column(Integer)
    first_dt = Column(DateTime)
    first_camp = Column(String)


class User_location(Base):
    __tablename__ = "users_location"
    user_id = Column(Integer, primary_key=True)
    location = Column(String)
    parsed_location = Column(String)
    country = Column(String)


class User_Profile(Base):
    __tablename__ = "user_profile"
    user_id = Column(Integer, primary_key=True)
    location = Column(String)
    parsed_location = Column(String)
    country = Column(String)
    age = Column(Integer)
    gender = Column(String)


class Bot_User(Base):
    __tablename__ = "bot_users"
    # tweet_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, primary_key=True)
    tweet_id = Column(Integer)
    first_dt = Column(DateTime)
    first_camp = Column(String)


class Hashtag(Base):
    __tablename__ = "hashtags"
    hashtag = Column(Text, primary_key=True)
    update_dt = Column(DateTime)
    count = Column(Integer)
    M_count = Column(Integer)
    K_count = Column(Integer)


class Hashtag75(Base):
    """
    history
    """
    __tablename__ = "hashtags75"
    hashtag = Column(Text, primary_key=True)
    update_dt = Column(DateTime)
    count = Column(Integer)


class Last_Week_Hashtag75(Base):
    __tablename__ = "last_week_hashtags75"
    hashtag = Column(Text, primary_key=True)
    update_dt = Column(DateTime)
    count = Column(Integer)


class Camp_Hashtag(Base):
    __tablename__ = "camp_hashtags"
    hashtag = Column(Text, primary_key=True)
    update_dt = Column(DateTime)
    camp = Column(String(10))


class Stat(Base):
    __tablename__ = "stat"
    dt = Column(DateTime, primary_key=True)
    tweet_count = Column(Integer)
    user_count = Column(Integer)
    tweet_cum_count = Column(Integer)
    cla_tweet_cum_count = Column(Integer)
    user_cum_count = Column(Integer)
    cla_user_cum_count = Column(Integer)
    K_tweet_count = Column(Integer)
    M_tweet_count = Column(Integer)
    U_tweet_count = Column(Integer)
    K_user_count = Column(Integer)
    M_user_count = Column(Integer)
    U_user_count = Column(Integer)
    I_user_count = Column(Integer)


class Bot_Stat(Base):
    __tablename__ = "bot_stat"
    dt = Column(DateTime, primary_key=True)
    tweet_count = Column(Integer)
    user_count = Column(Integer)
    tweet_cum_count = Column(Integer)
    cla_tweet_cum_count = Column(Integer)
    user_cum_count = Column(Integer)
    cla_user_cum_count = Column(Integer)
    K_tweet_count = Column(Integer)
    M_tweet_count = Column(Integer)
    U_tweet_count = Column(Integer)
    K_user_count = Column(Integer)
    M_user_count = Column(Integer)
    U_user_count = Column(Integer)
    I_user_count = Column(Integer)


class Daily_Predict(Base):
    __tablename__ = "daily_predict"
    dt = Column(DateTime, primary_key=True)
    U_Cristina = Column(Integer)
    U_Macri = Column(Integer)
    U_unclassified = Column(Integer)
    U_irrelevant = Column(Integer)


class History_Predict(Base):
    __tablename__ = "history_predict"
    dt = Column(DateTime, primary_key=True)
    U_Cristina = Column(Integer)
    U_Macri = Column(Integer)
    U_unclassified = Column(Integer)
    U_irrelevant = Column(Integer)


class Weekly_Predict(Base):
    __tablename__ = "weekly_predict"
    dt = Column(DateTime, primary_key=True)
    U_Cristina = Column(Integer)
    U_Macri = Column(Integer)
    U_unclassified = Column(Integer)
    U_irrelevant = Column(Integer)


class Percent(Base):
    __tablename__ = "percent"
    dt = Column(DateTime, primary_key=True)
    K = Column(Integer)
    M = Column(Integer)


class NoPASO_Predict(Base):
    __tablename__ = "nopaso_predict"
    dt = Column(DateTime, primary_key=True)
    U_Cristina = Column(Integer)
    U_Macri = Column(Integer)
    U_unclassified = Column(Integer)
    U_irrelevant = Column(Integer)


class Day3_Predict(Base):
    __tablename__ = "day3_predict"
    dt = Column(DateTime, primary_key=True)
    U_Cristina = Column(Integer)
    U_Macri = Column(Integer)
    U_unclassified = Column(Integer)
    U_irrelevant = Column(Integer)


class Day7_Predict(Base):
    __tablename__ = "day7_predict"
    dt = Column(DateTime, primary_key=True)
    U_Cristina = Column(Integer)
    U_Macri = Column(Integer)
    U_unclassified = Column(Integer)
    U_irrelevant = Column(Integer)


class Day14_Predict(Base):
    __tablename__ = "day14_predict"
    dt = Column(DateTime, primary_key=True)
    U_Cristina = Column(Integer)
    U_Macri = Column(Integer)
    U_unclassified = Column(Integer)
    U_irrelevant = Column(Integer)


class Day30_Predict(Base):
    __tablename__ = "day30_predict"
    dt = Column(DateTime, primary_key=True)
    U_Cristina = Column(Integer)
    U_Macri = Column(Integer)
    U_unclassified = Column(Integer)
    U_irrelevant = Column(Integer)


class Day60_Predict(Base):
    __tablename__ = "day60_predict"
    dt = Column(DateTime, primary_key=True)
    U_Cristina = Column(Integer)
    U_Macri = Column(Integer)
    U_unclassified = Column(Integer)
    U_irrelevant = Column(Integer)


class Bot_Weekly_Predict(Base):
    __tablename__ = "bot_weekly_predict"
    dt = Column(DateTime, primary_key=True)
    U_Cristina = Column(Integer)
    U_Macri = Column(Integer)
    U_unclassified = Column(Integer)
    U_irrelevant = Column(Integer)


class Weekly_Predict3(Base):
    __tablename__ = "weekly_predict3"
    dt = Column(DateTime, primary_key=True)
    U_Cristina = Column(Integer)
    U_Macri = Column(Integer)
    U_Massa = Column(Integer)
    U_unclassified = Column(Integer)
    U_irrelevant = Column(Integer)


class Other_Poll(Base):
    __tablename__ = "other_poll"
    id = Column(Integer, autoincrement=True, primary_key=True)
    dt = Column(String)
    name = Column(String)
    K = Column(Float)
    M = Column(Float)
    U = Column(Float)

# ------------- class definition ove -------------


def normalize_lower(text):
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode().lower()


def get_hashtags_from_tweet(_hashtags):
    if _hashtags:
        return ",".join(list(set([normalize_lower(tag["text"]) for tag in _hashtags])))
    else:
        return None


def get_source_text(_source):
    _sou = BeautifulSoup(_source, features="lxml").get_text()
    if _sou in official_twitter_clients:
        return None
    else:
        return _sou


def get_source_cnt(sess):
    from collections import Counter
    source_cnt = Counter()
    tweets = sess.query(Tweet.source).yield_per(1000)
    for t in tqdm(tweets):
        source_cnt[t[0]] += 1
    json.dump(source_cnt.most_common(), open(
        "data/source_cnt.json", "w"), indent=2)


def get_last_week():
    now = pendulum.now()
    dt = pendulum.DateTime(now.year, now.month, now.day)
    end = dt.add(days=-(dt.weekday-1))
    start = end.add(days=-7)
    # start <= dt < end
    return start, end


def get_all_tweets_id(sess):
    print("Get all tweets id ...")
    sess = get_session()
    tids_set = {t[0] for t in sess.query(Tweet.tweet_id).yield_per(5000)}
    print('have:', len(tids_set))
    sess.close()
    return tids_set


def get_tweets_json():
    set_tweets = set()
    # target_dir = ["201902", "201903", "201904", "201905"]
    target_dir = ["201907"]
    for _dir in target_dir:
        for in_name in os.listdir("disk/" + _dir):
            if in_name.endswith("PRO.txt") or in_name.endswith("Moreno.txt") or in_name.endswith("Sola.txt") \
                    or in_name.endswith("PASO.txt") or in_name.endswith("Rossi.txt") or in_name.endswith("elecciones.txt"):
                continue
            in_name = "disk/" + _dir + "/" + in_name
            print(in_name)
            for line in open(in_name):
                d = json.loads(line.strip())
                tweet_id = d["id"]
                if tweet_id in set_tweets:
                    continue
                set_tweets.add(tweet_id)
                yield d, tweet_id


def count_per_day():
    """
    用于测试是否数据量在下降
    """
    from collections import Counter
    tweet_cnt = Counter()
    for d, tid in tqdm(get_tweets_json()):
        dt = pendulum.from_format(
            d["created_at"], 'ddd MMM DD HH:mm:ss ZZ YYYY').format("MMDD")
        tweet_cnt[dt] += 1
    print(tweet_cnt.most_common())


def get_nopaso_tweets_json(paso_set):
    set_tweets = set()
    target_dir = ["201902", "201903", "201904", "201905"]
    for _dir in target_dir:
        for in_name in os.listdir("disk/" + _dir):
            if in_name.endswith("PRO.txt") or in_name.endswith("Moreno.txt") or in_name.endswith("Sola.txt"):
                continue
            if in_name.endswith("mauriciomacri OR PASO OR macrismo OR kirchnerismo OR peronismo.txt"):
                continue
            in_name = "disk/" + _dir + "/" + in_name
            print(in_name)
            for line in open(in_name, encoding="utf-8"):
                d = json.loads(line.strip())
                tweet_id = d["id"]
                if tweet_id in set_tweets:
                    continue
                set_tweets.add(tweet_id)

                if tweet_id in paso_set:
                    paso_set.remove(tweet_id)
    print("final:", len(paso_set))


def get_paso_tweets_id():
    set_tweets = set()
    target_dir = ["201905", "201906"]
    for _dir in target_dir:
        for in_name in os.listdir("disk/" + _dir):
            if not in_name.endswith("PASO.txt"):
                continue
            in_name = "disk/" + _dir + "/" + in_name
            print(in_name)
            for line in tqdm(open(in_name)):
                d = json.loads(line.strip())
                tweet_id = d["id"]
                if tweet_id in set_tweets:
                    continue
                set_tweets.add(tweet_id)

    return set_tweets


def count_paso():
    paso = set()
    nopaso = set()
    other = set()
    camp_query = ["mauriciomacri", "macrismo", "kirchnerismo", "peronismo"]
    target_dir = ["201902", "201903", "201904", "201905"]

    set_tweets = set()
    for _dir in target_dir:
        for in_name in os.listdir("disk/" + _dir):
            if not in_name.endswith("mauriciomacri OR PASO OR macrismo OR kirchnerismo OR peronismo.txt"):
                continue
            in_name = "disk/" + _dir + "/" + in_name
            print(in_name)
            for line in tqdm(open(in_name)):
                d = json.loads(line.strip())
                tweet_id = d["id"]
                if tweet_id in set_tweets:
                    continue
                set_tweets.add(tweet_id)

                text = normalize_lower(d["text"]).replace(
                    "\n", " ").replace("\t", " ")
                words = text.split()

                is_paso = False
                is_nopaso = False
                for w in words:
                    if "paso" in w:
                        paso.add(tweet_id)
                        is_paso = True
                    elif camp_query[0] in w or camp_query[1] in w or camp_query[2] in w or camp_query[3] in w:
                        nopaso.add(tweet_id)
                        is_nopaso = True
                if not is_paso and not is_nopaso:
                    other.add(tweet_id)

    with open("data/paso_id.txt", "w") as f:
        for tid in paso:
            f.write(str(tid) + "\n")

    with open("data/other_id.txt", "w") as f:
        for tid in nopaso:
            f.write(str(tid) + "\n")

    with open("data/nono_id.txt", "w") as f:
        for tid in other:
            f.write(str(tid) + "\n")

    print("final:", len(paso), len(nopaso), len(other))


def count_file_hashtag(in_file):
    from collections import Counter
    hashtags = Counter()
    for line in tqdm(open(in_file)):
        d = json.loads(line.strip())
        if d["hashtags"]:
            hts = get_hashtags_from_tweet(d["hashtags"]).split(",")
            for ht in hts:
                hashtags[ht] += 1

    with open("data/count_hashtags.txt", "w") as f:
        for ht in hashtags.most_common():
            f.write(f"{ht[0]},{ht[1]}\n")


def count_paso_camp():
    from TwProcess import load_models, bag_of_words_and_bigrams
    import pandas as pd

    tokenizer, v2, clf2 = load_models()

    X = []
    rst = []
    for line in tqdm(open("disk/201905/201905-PASO.txt")):
        d = json.loads(line.strip())
        words = bag_of_words_and_bigrams(tokenizer.tokenize(d["text"]))
        X.append(words)
        if len(X) == 5000:
            y = clf2.predict_proba(v2.transform(X))
            for i in range(len(y)):
                proM = round(y[i][1], 4)
                rst.append(proM)

            X = []

    pd.Series(rst).to_csv("disk/data/PASO_proba.csv")


def tweets_to_retweets(sess, start, end, clear=False):
    """
    导入转发推特
    """
    if clear:
        print("deleting >=", start, "<", end)
        sess.query(Retweet).filter(
            Retweet.dt >= start, Retweet.dt < end).delete()
        sess.commit()

    tweets_data = []
    for d, dt in read_end_file_for_retweets(start, end):
        if "in_reply_to_status_id" in d:
            continue
        elif "quoted_status_id" in d:
            continue
        elif "retweeted_status" in d:
            tid = d["id"]
            uid = d["user"]["id"]
            o_tid = d["retweeted_status"]["id"]
            o_uid = d["retweeted_status"]["user"]["id"]

            if sess.query(exists().where(Tweet.tweet_id == tid)).scalar():
                tweets_data.append(
                    Retweet(
                        tweet_id=tid,
                        dt=dt,
                        user_id=uid,
                        ori_tweet_id=o_tid,
                        ori_user_id=o_uid
                    )
                )

        if len(tweets_data) == 5000:
            sess.add_all(tweets_data)
            sess.commit()
            tweets_data = []

    if tweets_data:
        sess.add_all(tweets_data)
        sess.commit()


def tweets_to_retweets_all(sess):
    """
    导入转发推特
    """
    tweets_data = []
    for d in read_all_retweets():
        if "in_reply_to_status_id" in d:
            continue
        elif "quoted_status_id" in d:
            continue
        elif "retweeted_status" in d:
            tid = d["id"]
            uid = d["user"]["id"]
            o_tid = d["retweeted_status"]["id"]
            o_uid = d["retweeted_status"]["user"]["id"]
            dt = pendulum.from_format(d["created_at"], 'ddd MMM DD HH:mm:ss ZZ YYYY')
            tweets_data.append(
                Retweet(
                    tweet_id=tid,
                    dt=dt,
                    user_id=uid,
                    ori_tweet_id=o_tid,
                    ori_user_id=o_uid
                )
            )

        if len(tweets_data) == 5000:
            sess.add_all(tweets_data)
            sess.commit()
            tweets_data = []

    if tweets_data:
        sess.add_all(tweets_data)
        sess.commit()


def tweets_to_db(sess):
    """
    导入全部数据，并分类
    """
    from TwProcess import load_models, bag_of_words_and_bigrams

    tokenizer, v2, clf2 = load_models()

    X = []
    tweets_data = []
    tweets_json = get_tweets_json()
    for d, tweet_id in tweets_json:
        uid = d["user"]["id"]
        dt = pendulum.from_format(
            d["created_at"], 'ddd MMM DD HH:mm:ss ZZ YYYY')

        _sou = get_source_text(d["source"])
        hts = get_hashtags_from_tweet(d["hashtags"])

        tweets_data.append(
            Tweet(tweet_id=tweet_id, user_id=uid, dt=dt, source=_sou, hashtags=hts))

        words = bag_of_words_and_bigrams(tokenizer.tokenize(d["text"]))
        X.append(words)

        if len(tweets_data) == 5000:
            # y = clf1.predict_proba(v1.transform(X))
            # for i in range(len(y)):
            #     proK = round(y[i][0], 4)
            #     proM = round(y[i][1], 4)
            #     proA = round(y[i][2], 4)
            #     tweets_data[i].proK3 = proK
            #     tweets_data[i].proM3 = proM
            #     tweets_data[i].proA3 = proA
                # print(y[i])

            y = clf2.predict_proba(v2.transform(X))
            for i in range(len(y)):
                # print(y.shape)
                proK = round(y[i][0], 4)
                proM = round(y[i][1], 4)
                tweets_data[i].proK = proK
                tweets_data[i].proM = proM

            sess.add_all(tweets_data)
            sess.commit()
            X = []
            tweets_data = []

    if tweets_data:
        # y = clf1.predict_proba(v1.transform(X))
        # for i in range(len(y)):
        #     proK = round(y[i][0], 4)
        #     proM = round(y[i][1], 4)
        #     proA = round(y[i][2], 4)
        #     tweets_data[i].proK3 = proK
        #     tweets_data[i].proM3 = proM
        #     tweets_data[i].proA3 = proA

        y = clf2.predict_proba(v2.transform(X))
        for i in range(len(y)):
            proK = round(y[i][0], 4)
            proM = round(y[i][1], 4)
            tweets_data[i].proK = proK
            tweets_data[i].proM = proM

        sess.add_all(tweets_data)
        sess.commit()


def tweets_to_source(sess):
    """
    导入全部数据，并分类
    """
    tweets_data = []
    tweets_json = get_tweets_json()
    for d, tweet_id in tweets_json:
        uid = d["user"]["id"]
        _sou = get_source_text(d["source"])
        tweets_data.append(
            Source(tweet_id=tweet_id, user_id=uid, source=_sou))

        if len(tweets_data) == 5000:
            sess.add_all(tweets_data)
            sess.commit()
            tweets_data = []

    if tweets_data:
        sess.add_all(tweets_data)
        sess.commit()


def read_end_file(start, end):
    set_tweets = set()
    target_dir = ["201903", "201904", "201905", "201906", "201907", "201908", "201909", "201910"]

    from deal_with_Queries import File_Checker
    checker = File_Checker()
    #? Sha qing kuang?
    for _dir in target_dir:
        # print("Dir >", _dir)
        for in_name in os.listdir("disk/" + _dir):
            # ignore
            if checker.ignore_it(in_name):
                # print("~Ignore:", in_name)
                continue
            print(in_name, "start ...")
            in_name = "disk/" + _dir + "/" + in_name
            cnt = 0

            # exists_cnt = 0
            with FileReadBackwards(in_name) as f:
                while True:
                    line = f.readline()
                    if not line:
                        print(cnt, "end!")
                        print("-" * 80)
                        break

                    d = json.loads(line.strip())
                    tweet_id = d["id"]
                    if tweet_id in set_tweets:
                        continue
                    set_tweets.add(tweet_id)

                    dt = pendulum.from_format(
                        d["created_at"], 'ddd MMM DD HH:mm:ss ZZ YYYY')
                    if dt < start:
                        print("sum:", cnt, d["created_at"], "end!")
                        break
                    if dt >= end:
                        continue

                    if cnt % 50000 == 0:
                        print("New data comming ->", cnt)
                    cnt += 1
                    yield d, dt


def read_end_file_for_retweets(start, end):

    set_tweets = set()
    target_dir = ["201905", "201906", "201907", "201908",
                  "201909", "201910"]

    from deal_with_Queries import File_Checker
    checker = File_Checker()
    #? Sha qing kuang?
    for _dir in target_dir:
        # print("Dir >", _dir)
        for in_name in os.listdir("disk/" + _dir):
            # ignore
            if checker.ignore_it(in_name):
                # print("~Ignore:", in_name)
                continue
            print(in_name, "start ...")
            in_name = "disk/" + _dir + "/" + in_name
            cnt = 0
            # exists_cnt = 0
            with FileReadBackwards(in_name) as f:
                while True:
                    line = f.readline()
                    if not line:
                        print("end of", in_name, cnt)
                        break

                    d = json.loads(line.strip())
                    tweet_id = d["id"]
                    if tweet_id in set_tweets:
                        continue
                    set_tweets.add(tweet_id)

                    dt = pendulum.from_format(
                        d["created_at"], 'ddd MMM DD HH:mm:ss ZZ YYYY')
                    if dt < start:
                        print("sum:", cnt, d["created_at"], "end!")
                        break
                    if dt >= end:
                        continue

                    if cnt % 50000 == 0:
                        print("New data:", cnt)
                    cnt += 1
                    yield d, dt


def read_all_retweets():
    set_tweets = set()
    in_names = [
        "201902", "201903", "201904",
        "201905", "201906", "201907", 
        "201908", "201909", "201910"
    ]
    for in_name in in_names:
        print(in_name)
        for line in open("D:\\ARG2019\\raw_tweets\\" + in_name + ".lj", encoding="utf8"):
            d = json.loads(line.strip())
            if d["id"] not in set_tweets:
                set_tweets.add(d["id"])
                yield d


def read_end_file_PASO(start, end):

    import random
    set_tweets = set()
    # set_users = set()

    # target_dir = ["201902", "201903", "201904"]
    target_dir = ["201906"]

    for _dir in target_dir:
        for in_name in tqdm(os.listdir("disk/" + _dir)):
            if in_name.endswith("PRO.txt") or in_name.endswith("Moreno.txt") \
                    or in_name.endswith("Sola.txt"):
                continue

            print(in_name, "start ...")
            in_name = "disk/" + _dir + "/" + in_name
            cnt = 0
            # exists_cnt = 0
            with FileReadBackwards(in_name) as f:
                while True:
                    line = f.readline()
                    if not line:
                        print("End of file!", in_name, cnt)
                        break

                    if in_name.endswith("PASO.txt"):
                        God = random.random()
                        if start.to_date_string() == "2019-06-09" and God > 0.8:
                            continue
                        elif start.to_date_string() == "2019-06-10" and God > 0.6:
                            continue
                        elif start.to_date_string() == "2019-06-11" and God > 0.4:
                            continue
                        elif start.to_date_string() == "2019-06-12" and God > 0.2:
                            continue

                    d = json.loads(line.strip())
                    tweet_id = d["id"]
                    if tweet_id in set_tweets:
                        continue
                    set_tweets.add(tweet_id)

                    dt = pendulum.from_format(
                        d["created_at"], 'ddd MMM DD HH:mm:ss ZZ YYYY')
                    if dt < start:
                        print("sum:", cnt, d["created_at"], "end!")
                        break
                    if dt >= end:
                        continue

                    if cnt % 50000 == 0:
                        print("New data:", cnt)
                    cnt += 1
                    yield d, dt


def tweets_to_db_v2(sess, start, end, clear=False):
    """
    import tweets to database with prediction
    """
    if clear:
        print("deleting >=", start, "<", end)
        sess.query(Tweet).filter(Tweet.dt >= start, Tweet.dt < end).delete()
        sess.commit()

    from classifier import Camp_Classifier
    Lebron = Camp_Classifier()
    Lebron.load2()

    X = []
    tweets_data = []

    for d, dt in read_end_file(start, end):
        tweet_id = d["id"]
        uid = d["user"]["id"]
        _sou = get_source_text(d["source"])
        hts = get_hashtags_from_tweet(d["hashtags"])

        tweets_data.append(
            Tweet(tweet_id=tweet_id, user_id=uid,
                  dt=dt, source=_sou, hashtags=hts)
        )

        X.append(d)
        if len(tweets_data) == 2000:
            json_rst = Lebron.predict(X)
            for i in range(len(tweets_data)):
                tweets_data[i].proK = json_rst[tweets_data[i].tweet_id][0]
                tweets_data[i].proM = json_rst[tweets_data[i].tweet_id][1]

            sess.add_all(tweets_data)
            sess.commit()
            X = []
            tweets_data = []

    if tweets_data:
        json_rst = Lebron.predict(X)
        for i in range(len(tweets_data)):
            tweets_data[i].proK = json_rst[tweets_data[i].tweet_id][0]
            tweets_data[i].proM = json_rst[tweets_data[i].tweet_id][1]

        sess.add_all(tweets_data)
        sess.commit()


def tweets_to_db_v3(sess, start, end, clear=False):
    """
    import tweets to database with prediction
    """
    if clear:
        print("deleting >=", start, "<", end)
        sess.query(Tweet).filter(Tweet.dt >= start, Tweet.dt < end).delete()
        sess.commit()

    from classifier import Camp_Classifier
    Lebron = Camp_Classifier()
    Lebron.load2()

    X = []
    tweets_data = []

    for d, dt in read_end_file(start, end):
        tweet_id = d["id"]
        uid = d["user"]["id"]
        _sou = get_source_text(d["source"])
        hts = get_hashtags_from_tweet(d["hashtags"])

        tweets_data.append(
            Tweet(tweet_id=tweet_id, user_id=uid,
                  dt=dt, source=_sou, hashtags=hts)
        )

        X.append(d)
        if len(tweets_data) == 2000:
            json_rst = Lebron.predict(X)
            for i in range(len(tweets_data)):
                tweets_data[i].proK = json_rst[tweets_data[i].tweet_id][0]
                tweets_data[i].proM = json_rst[tweets_data[i].tweet_id][1]

            sess.add_all(tweets_data)
            sess.commit()
            X = []
            tweets_data = []

    if tweets_data:
        json_rst = Lebron.predict(X)
        for i in range(len(tweets_data)):
            tweets_data[i].proK = json_rst[tweets_data[i].tweet_id][0]
            tweets_data[i].proM = json_rst[tweets_data[i].tweet_id][1]

        sess.add_all(tweets_data)
        sess.commit()

        
def tweets_to_db_train():
    """
    导入训练数据

    每段时间满足某组hashtag规则
    不能是转发文本
    """
    
    def get_train_tweets(self, all_id):
        set_tweets = set()
        #* 需要修改！！
        # target_dir = ["201908"]
        if self.now == "201904":
            target_dir = ["201902", "201903"]
        else:
            target_dir = [pendulum.parse(self.now + "01").add(days=-1).format("YYYYMM")]

        print("Targets:", target_dir)

        from deal_with_Queries import File_Checker
        checker = File_Checker()
        #? Sha qing kuang?
        for _dir in target_dir:
            # print("Dir >", _dir)
            for in_name in os.listdir("disk/" + _dir):
                # ignore
                if checker.ignore_it(in_name):
                    # print("~Ignore:", in_name)
                    continue
                in_name = "disk/" + _dir + "/" + in_name
                # print(in_name)
                for line in open(in_name, encoding="utf-8"):
                    d = json.loads(line.strip())
                    tweet_id = d["id"]
                    if tweet_id in all_id and tweet_id not in set_tweets:
                        if 'retweeted_status' in d and d["text"].startswith("RT @"): # ignoring retweets
                            continue
                        # dt = pendulum.from_format(d["created_at"],
                        #     'ddd MMM DD HH:mm:ss ZZ YYYY').to_date_string()
                        text = d["text"].replace("\n", " ").replace("\t", " ")
                        set_tweets.add(tweet_id)

                        yield tweet_id, text


    def get_train_data(self):
        now = self.now
        K_ht, M_ht, A_ht, all_hts = self.K_ht, self.M_ht, self.A_ht, self.hts
        with open(f"disk/data/{now}/hts.mod", "w") as f:
            for ht in K_ht:
                f.write(f"K\t{ht}\n")
            for ht in M_ht:
                f.write(f"M\t{ht}\n")
            # for ht in A_ht:
            #     f.write(f"L\t{ht}\n")

        K_id = set()
        M_id = set()
        # A_id = set()
        # K_A_id = set()

        sess = get_session()
        # tweets = get_all_tweets_with_hashtags(sess)
        #* 需要修改！！
        if self.now == "201904":
            start = pendulum.datetime(2019, 2, 1, tz="UTC") # include this date
        else:
            start = pendulum.parse(self.now + "01", tz="UTC").add(days=-1)
        end = pendulum.parse(self.now + "01", tz="UTC")
        # end = pendulum.datetime(2019, 8, 1, tz="UTC") # not include this date

        print("Getting tweets with hashtags", start, end)
        tweets = get_tweets_with_hashtags(sess, start, end)
        for t in tqdm(tweets):
            hashtags = t[1].split(",")
            K_bingo = False
            M_bingo = False
            R_bingo = False
            # A_bingo = False

            # consider bingo_nums == 1
            for ht in hashtags:
                if ht in self.remove_hts:
                    R_bingo = True
                    break
                elif ht in K_ht:
                    K_bingo = True
                elif ht in M_ht:
                    M_bingo = True

            # dt = pendulum.instance(t[2]).to_date_string()
            # print(dt)
            if R_bingo:
                continue
            elif K_bingo and not M_bingo:
                K_id.add(int(t[0]))
            elif M_bingo and not K_bingo:
                M_id.add(int(t[0]))

            # if bingo_nums == 2:
            #     if K_bingo and A_bingo:
            #         K_A_id.add(t[0])

        print("Number of tweets with hashtags:", len(K_id), len(M_id))
        sess.close()

        # all_id = K_id | M_id
        # all_id = K_id | M_id | A_id | K_A_id
        # print(len(K_id), len(M_id), len(all_id))

        # from collections import defaultdict
        # K_c = defaultdict(int)
        # M_c = defaultdict(int)
        # KA_c = defaultdict(int)

        # json.dump(K_c, open("data/Kc.txt", "w"), indent=1)
        # json.dump(M_c, open("data/Mc.txt", "w"), indent=1)
        # json.dump(A_c, open("data/Ac.txt", "w"), indent=1)
        # json.dump(KA_c, open("data/KAc.txt", "w"), indent=1)

        all_id = K_id | M_id
        K_bingo_file = open(f"disk/data/{now}/K.txt", "w")
        M_bingo_file = open(f"disk/data/{now}/M.txt", "w")
        for t in self.get_train_tweets(all_id):
            _id, text = t
            if _id in K_id:
                K_bingo_file.write(text + "\n")
            elif _id in M_id:
                M_bingo_file.write(text + "\n")

        K_bingo_file.close()
        M_bingo_file.close()

        with open(f"disk/data/{now}/train.txt", "a") as f:
            for line in open(f"disk/data/{now}/K.txt"):
                f.write(f"0\t{line}")
            for line in open(f"disk/data/{now}/M.txt"):
                f.write(f"1\t{line}")
 
        # with open(f"disk/data/traindata-{now}-3.txt", "w") as f:
        #     for line in open(f"disk/data/K-{now}.txt"):
        #         f.write(f"0\t{line}")
        #     for line in open(f"disk/data/M-{now}.txt"):
        #         f.write(f"1\t{line}")
        #     for line in open(f"disk/data/L-{now}.txt"):
        #         f.write(f"2\t{line}")


def tweets_to_db_v3(sess, start, end, clear=False):
    """
    导入数据，并3分类
    """
    if clear:
        print("deleting >=", start, "<", end)
        sess.query(Tweet3).filter(Tweet3.dt >= start, Tweet3.dt < end).delete()
        sess.commit()

    from classify_camp import Classifer
    classifier = Classifer()
    classifier.load3()
    X = []
    tweets_data = []

    for d, dt in read_end_file(start, end):
        tweet_id = d["id"]

        tweets_data.append(
            Tweet3(tweet_id=tweet_id, dt=dt)
        )

        X.append(d)
        if len(tweets_data) == 5000:
            json_rst = classifier.predict3(X)
            for i in range(len(tweets_data)):
                tweets_data[i].proA3 = json_rst[tweets_data[i].tweet_id]
            tweets_data = [_t for _t in tweets_data if _t.proA3 > 0]
            sess.add_all(tweets_data)
            sess.commit()
            X = []
            tweets_data = []

    if tweets_data:
        json_rst = classifier.predict3(X)
        for i in range(len(tweets_data)):
            tweets_data[i].proA3 = json_rst[tweets_data[i].tweet_id]
        tweets_data = [_t for _t in tweets_data if _t.proA3 > 0]
        sess.add_all(tweets_data)
        sess.commit()


########################## 我是天才 ##########################
def db_to_users(sess, start, end, bots=False):
    # 只获取一次
    users = {}  # 还没插入，可以随时改
    exist_users = get_all_users(sess, bots=bots)
    tweets = get_tweets(sess, start, end, bots=bots)

    for t in tqdm(tweets):
        uid = t.user_id
        tid = t.tweet_id
        camp = None
        if t.proM >= 0.75:
            camp = "M"
        elif t.proM < 0.25:
            camp = "K"

        t_dt = t.dt
        if uid in exist_users:
            continue

        if uid not in users:
            users[uid] = [tid, t_dt, camp]
        elif tid < users[uid][0]:
            users[uid] = [tid, t_dt, camp]

    if bots:
        users = [Bot_User(user_id=uid, tweet_id=v[0],
                          first_dt=v[1], first_camp=v[2]) for uid, v in users.items()]
    else:
        users = [User(user_id=uid, tweet_id=v[0],
                      first_dt=v[1], first_camp=v[2]) for uid, v in users.items()]

    print(f"adding {len(users)} users ...")
    sess.add_all(users)
    sess.commit()


###################### hashtags ######################
def tweets_db_to_hashtags(sess, start, end):
    """
    One month
    """
    from collections import defaultdict
    _hashtags = defaultdict(int)
    _ht_M = defaultdict(int)
    _ht_K = defaultdict(int)

    tweets = sess.query(Tweet.hashtags).filter(
        Tweet.source.is_(None),
        Tweet.hashtags.isnot(None),
        Tweet.proM > 0.75,
        Tweet.dt >= start,
        Tweet.dt < end).yield_per(5000)

    for t in tqdm(tweets):
        hts = t[0].split(",")
        for ht in hts:
            _hashtags[ht] += 1
            _ht_M[ht] += 1

    tweets = sess.query(Tweet.hashtags).filter(
        Tweet.source.is_(None),
        Tweet.hashtags.isnot(None),
        Tweet.dt >= start,
        Tweet.proK > 0.75,
        Tweet.dt < end).yield_per(5000)

    for t in tqdm(tweets):
        hts = t[0].split(",")
        for ht in hts:
            _hashtags[ht] += 1
            _ht_K[ht] += 1

    end = pendulum.today()
    _hashtags = [Hashtag(hashtag=ht, update_dt=end, count=cnt, M_count=_ht_M[ht], K_count=_ht_K[ht])
                 for ht, cnt in _hashtags.items()]
    print(len(_hashtags))

    sess.query(Hashtag).delete()
    sess.commit()
    sess.add_all(_hashtags)
    sess.commit()


def get_top_hashtags(sess):
    from collections import Counter
    _hashtags = Counter()

    tweets = sess.query(Tweet.hashtags).filter(
        Tweet.source.is_(None),
        Tweet.hashtags.isnot(None),
        Tweet.dt >= "2019-04-10",
        Tweet.dt < "2019-05-24").yield_per(5000)

    for t in tqdm(tweets):
        hts = t[0].split(",")
        for ht in hts:
            _hashtags[ht] += 1

    print(_hashtags.most_common(200))


def tweets_db_to_hashtags75(sess, end):
    """
    all tweets
    """
    from collections import defaultdict
    _hashtags = defaultdict(int)

    tweets = sess.query(Tweet.hashtags).filter(
        Tweet.source.is_(None),
        Tweet.hashtags.isnot(None),
        Tweet.dt < end,
        or_(Tweet.proM > 0.75, Tweet.proK > 0.75)).yield_per(5000)

    for t in tqdm(tweets):
        hts = t[0].split(",")
        for ht in hts:
            _hashtags[ht] += 1

    _hashtags = [Hashtag75(hashtag=ht, update_dt=end, count=cnt)
                 for ht, cnt in _hashtags.items()]
    print(len(_hashtags))

    sess.query(Hashtag75).delete()
    sess.commit()

    sess.add_all(_hashtags)
    sess.commit()


def tweets_db_to_hashtags_KM(sess, end):
    from collections import defaultdict
    _hashtags = defaultdict(int)

    tweets = sess.query(Tweet.hashtags).filter(
        Tweet.source.is_(None),
        Tweet.hashtags.isnot(None),
        Tweet.dt < end, Tweet.proM >= 0.75).yield_per(5000)

    for t in tqdm(tweets):
        hts = t[0].split(",")
        for ht in hts:
            _hashtags[ht] += 1

    _hashtags = defaultdict(int)
    tweets = sess.query(Tweet.hashtags).filter(
        Tweet.source.is_(None),
        Tweet.hashtags.isnot(None),
        Tweet.dt < end, Tweet.proM >= 0.75).yield_per(5000)

    for t in tqdm(tweets):
        hts = t[0].split(",")
        for ht in hts:
            _hashtags[ht] += 1


def update_hashtags75(sess, end):
    from collections import defaultdict
    _hashtags = defaultdict(int)

    tweets = sess.query(Tweet.hashtags, Tweet.proM).filter(
        and_(Tweet.source.is_(None),
             and_(Tweet.hashtags.isnot(None),
                  and_(Tweet.dt < end)))).yield_per(5000)
    for t in tqdm(tweets):
        if t[1] > 0.75 or t[1] < 0.25:
            hts = t[0].split(",")
            for ht in hts:
                _hashtags[ht] += 1

    _hashtags = [Hashtag75(hashtag=ht, update_dt=end, count=cnt)
                 for ht, cnt in _hashtags.items()]
    print(len(_hashtags))

    sess.query(Hashtag75).delete()
    sess.commit()
    sess.add_all(_hashtags)
    sess.commit()


def count_of_hashtags(sess, start, end):
    from collections import defaultdict
    _hashtags = defaultdict(int)
    period = pendulum.period(start, end)

    for dt in period:
        tweets = get_tweets_day_with_hashtags(sess, dt)
        for t in tqdm(tweets):
            if t.proM > 0.75 or t.proM < 0.25:
                for ht in t.hashtags.split(","):
                    _hashtags[ht] += 1
    print(_hashtags)
    return dict(_hashtags)
    
    
def tweets_db_to_hashtags75_lastweek(sess, end):
    from collections import defaultdict
    _hashtags = defaultdict(int)
    start = end.add(days=-7)
    period = pendulum.period(start, end)

    for dt in period:
        tweets = get_tweets_day_with_hashtags(sess, dt)
        for t in tqdm(tweets):
            if t.proM > 0.75 or t.proK > 0.75:
                for ht in t.hashtags.split(","):
                    _hashtags[ht] += 1
    _hashtags = [Last_Week_Hashtag75(hashtag=ht, update_dt=end, count=cnt)
                 for ht, cnt in _hashtags.items()]
    print(len(_hashtags))

    sess.query(Last_Week_Hashtag75).delete()
    print("clear Last_Week_Hashtag75 ...")
    sess.commit()

    sess.add_all(_hashtags)
    sess.commit()


def get_hashtags75_v2(sess):
    from collections import Counter
    _hashtags = Counter()

    bingo_hashtags = [
        ('fernandezfernandez', 7535),
        ('ganafernandez', 2505),
        ('arrugocristina', 2200),
        ('ahora', 2033),
        ('cfk', 1896),
        ('lacornisa', 1674),
        ('elecciones2019', 1577),
        ('cristinasomostodos', 1546),
        ('unidad', 1522),
        ('buensabado', 1476),
        ('argentina', 1465),
        ('jujuy', 1295),
        ('salta', 1292),
        ('defensoresdelcambio', 1283),
        ('encuesta', 1225),
        ('cambiemos', 1208),
        ('lapampa', 1168),
        ('buendomingo', 1132),
        ('fernandezfernandez2019', 1022),
        ('haceminutos', 983),
        ('nsb', 981),
        ('lanochedeml', 958),
        ('26m', 913),
        ('politica', 877),
        ('albertofernandez', 877),
        ('porunaargentinamejor', 872),
        ('cristina', 871),
        ('aunestamosatiempo', 871),
        ('18may', 817),
        ('elecciones', 775),
        ('mauroenamerica', 668),
        ('venezuela', 665),
        ('convencionucrpba', 657),
        ('escontodos', 629),
        ('urgente', 578),
        ('macri', 530),
        ('4t', 508),
    ]

    set_hts = set([ht[0] for ht in bingo_hashtags])
    start = pendulum.datetime(2019, 5, 18)
    end = pendulum.datetime(2019, 5, 20)
    period = pendulum.period(start, end)
    M_SAT_MON = open("disk/data/M_SAT_MON.id", "w")
    K_SAT_MON = open("disk/data/K_SAT_MON.id", "w")

    for dt in period:
        print(dt)
        tweets = get_tweets_day_with_hashtags(sess, dt)
        for t in tqdm(tweets):
            _goal = False
            if t.proM > 0.75 or t.proK > 0.75:
                for ht in t.hashtags.split(","):
                    # _hashtags[ht] += 1
                    if ht in set_hts:
                        if t.proM > 0.75:
                            M_SAT_MON.write(str(t.tweet_id) + "\n")
                        if t.proM < 0.25:
                            K_SAT_MON.write(str(t.tweet_id) + "\n")
                        break


def get_raw_tweets():

    M_f = open("disk/data/M_SAT_MON.txt", "w")
    K_f = open("disk/data/K_SAT_MON.txt", "w")

    M_SAT_MON = set([int(line.strip())
                     for line in open("disk/data/M_SAT_MON.id")])
    K_SAT_MON = set([int(line.strip())
                     for line in open("disk/data/K_SAT_MON.id")])

    M_count, K_count = 0, 0

    for t, tweetid in get_tweets_json():
        if tweetid in M_SAT_MON:
            text = t["text"].replace("\n", " ").replace("\t", " ")
            M_f.write(f"id:{tweetid}\t{text}\n")
            M_count += 1
        if tweetid in K_SAT_MON:
            text = t["text"].replace("\n", " ").replace("\t", " ")
            K_f.write(f"id:{tweetid}\t{text}\n")
            K_count += 1


# ******************** Very important ******************** #
def db_to_stat_predict(sess, start, end, bots=False, clear=False):

    if clear:
        if bots:
            sess.query(Bot_Stat).filter(Bot_Stat.dt >=
                                        start, Bot_Stat.dt < end).delete()
        else:
            sess.query(Stat).filter(Stat.dt >= start, Stat.dt < end).delete()
        sess.commit()

    _dt = start
    while _dt < end:  # per day
        print(_dt)
        users_support = {}
        new_tweets_cnt = 0
        K_tweets, M_tweets, U_tweets = 0, 0, 0
        K_users, M_users, U_users, I_users = 0, 0, 0, 0

        if bots:
            tweets = get_bot_tweets_day(sess, _dt)
        else:
            tweets = get_tweets_day(sess, _dt)

        remove_uids = set()
        for t in tqdm(tweets):
            uid = t.user_id
            proM = t.proM
            if uid not in users_support:
                users_support[uid] = [0, 0, 0]  # K, M, unclassified
            new_tweets_cnt += 1

            if proM < 0:
                remove_uids.add(uid)
            elif proM >= 0.75:
                M_tweets += 1
                users_support[uid][1] += 1
            elif proM < 0.25:
                K_tweets += 1
                users_support[uid][0] += 1
            else:
                U_tweets += 1
                users_support[uid][2] += 1

        for u, _cla in users_support.items():
            if u in remove_uids or (_cla[0] == 0 and _cla[1] == 0):
                I_users += 1
            elif _cla[0] > _cla[1]:
                K_users += 1
            elif _cla[1] > _cla[0]:
                M_users += 1
            else:
                U_users += 1

        if bots:
            cum_t = sess.query(Tweet).filter(Tweet.dt < _dt.add(
                days=1), Tweet.source.isnot(None)).count()
            c_cum_t = sess.query(Tweet).filter(Tweet.dt < _dt.add(days=1), Tweet.source.isnot(None),
                                               or_(Tweet.proK > 0.75, Tweet.proM > 0.75)).count()

            cum_u = sess.query(Bot_User).filter(
                Bot_User.first_dt < _dt.add(days=1)).count()
            c_cum_u = sess.query(Bot_User).filter(Bot_User.first_dt < _dt.add(days=1),
                                                  Bot_User.first_camp.isnot(None)).count()
            sess.add(
                Bot_Stat(dt=_dt,
                         tweet_count=new_tweets_cnt, user_count=len(
                             users_support),
                         tweet_cum_count=cum_t, user_cum_count=cum_u,
                         cla_tweet_cum_count=c_cum_t, cla_user_cum_count=c_cum_u,
                         K_tweet_count=K_tweets, M_tweet_count=M_tweets, U_tweet_count=U_tweets,
                         K_user_count=K_users, M_user_count=M_users,
                         U_user_count=U_users, I_user_count=I_users,))
        else:
            cum_t = sess.query(Tweet).filter(Tweet.dt < _dt.add(
                days=1), Tweet.source.is_(None)).count()
            c_cum_t = sess.query(Tweet).filter(Tweet.dt < _dt.add(days=1), Tweet.source.is_(None),
                                               or_(Tweet.proK > 0.75, Tweet.proM > 0.75)).count()

            cum_u = sess.query(User).filter(
                User.first_dt < _dt.add(days=1)).count()
            c_cum_u = sess.query(User).filter(User.first_dt < _dt.add(days=1),
                                              User.first_camp.isnot(None)).count()

            # new_user_cnt = sess.query(User).filter(
            #     and_(User.first_dt >= _dt, User.first_dt < _dt.add(days=1))).count()

            _s = Stat(dt=_dt,
                      tweet_count=new_tweets_cnt, user_count=len(
                          users_support),

                      tweet_cum_count=cum_t, user_cum_count=cum_u,
                      cla_tweet_cum_count=c_cum_t, cla_user_cum_count=c_cum_u,

                      K_tweet_count=K_tweets, M_tweet_count=M_tweets, U_tweet_count=U_tweets,

                      K_user_count=K_users, M_user_count=M_users,
                      U_user_count=U_users, I_user_count=I_users,)

            pprint(_s.__dict__)
            sess.add(_s)

        sess.commit()
        _dt = _dt.add(days=1)


def db_to_stat_predict_v2(sess, start, end):

    out_file = open("data/stat_v3.json", "w")
    for _dt in pendulum.Period(start, end):
        print(_dt)
        users_support = {}
        new_tweets_cnt = 0
        K_tweets, M_tweets, U_tweets = 0, 0, 0
        K_users, M_users, U_users, I_users = 0, 0, 0, 0
        # tweets = get_bot_tweets_day(sess, _dt)
        tweets = get_tweets_day(sess, _dt)

        remove_uids = set()
        for t in tqdm(tweets):
            uid = t.user_id
            proM = t.proM
            if uid not in users_support:
                users_support[uid] = [0, 0, 0]  # K, M, unclassified
            new_tweets_cnt += 1

            if proM < 0:
                remove_uids.add(uid)
            elif proM >= 0.66:
                M_tweets += 1
                users_support[uid][1] += 1
            elif proM < 0.33:
                K_tweets += 1
                users_support[uid][0] += 1
            else:
                U_tweets += 1
                users_support[uid][2] += 1

        for u, _cla in users_support.items():
            if u in remove_uids:
                continue
            elif _cla[0] > _cla[1]:
                K_users += 1
            elif _cla[1] > _cla[0]:
                M_users += 1
            else:
                U_users += 1

        _s = dict(dt=_dt.to_date_string(),
                    tweet_count=new_tweets_cnt, classified_tweet_count=K_tweets + M_tweets,
                    user_count=K_users + M_users + U_users, classified_user_count=K_users + M_users,
                    K_tweet_count=K_tweets, M_tweet_count=M_tweets,
                    K_user_count=K_users, M_user_count=M_users)
        out_file.write(json.dumps(_s) + "\n")


def add_camp_hashtags(clear=False):
    sess = get_session()

    if clear:
        sess.query(Camp_Hashtag).delete()
        sess.commit()

    for line in open("data/hashtags/2019-09-05.txt"):
        # print(line)
        w = line.strip().split()
        ht = normalize_lower(w[1])
        d = Camp_Hashtag(hashtag=ht, update_dt=pendulum.datetime(
            2019, 6, 23), camp=w[0])
        try:
            sess.add(d)
            sess.commit()
            print("add:", w)
        except Exception as e:
            print(e)

    sess.close()
    get_camp_hashtags()


def add_other_polls(clear=False):
    sess = get_session()

    if clear:
        sess.query(Other_Poll).delete()
        sess.commit()

    import pandas as pd

    data = pd.read_csv("data/wiki-data.csv")

    for i, row in data.iterrows():
        print(row.poll_name, row.poll_dt)
        sess.add(Other_Poll(
            dt=row.poll_dt,
            name=row.poll_name,
            K=row.K / 100,
            M=row.M / 100,
            U=1 - row.K / 100 - row.M / 100
        ))

    sess.commit()
    sess.close()


# run it each day
def predict_day(sess, dt, lag=14, bots=False, clear=False):
    """
    use tweets in the last 14 (lag) days to predict everyday
    dt is today,
    so, start is -14 day, end is -1 day.
    save -1 day in the db
    """

    if clear:
        if bots:
            sess.query(Bot_Weekly_Predict).filter(
                Bot_Weekly_Predict.dt == dt).delete()
            sess.commit()
        elif not bots:
            sess.query(Weekly_Predict).filter(Weekly_Predict.dt == dt).delete()
            sess.commit()

    start = dt.add(days=-lag)
    end = dt

    users = {}
    # print("predict daily!", start, "~", end)
    tweets = get_tweets(sess, start, end, bots=bots)
    # remove_uid = set()

    for t in tweets:
        uid = t.user_id
        # if uid in remove_uid:
        # continue
        if uid not in users:
            users[uid] = {
                "proM": 0,
                "proK": 0,
                "Unclassified": 0,
                "Junk": 0,
            }

        if t.proM < 0:
            # remove_uid.add(uid)
            # if uid in users:
                # users.pop(uid)
            users[uid]["Junk"] += 1
        elif t.proM >= 0.75:
            users[uid]["proM"] += 1
        elif t.proM < 0.25:
            users[uid]["proK"] += 1
        else:
            users[uid]["Unclassified"] += 1

    cnt = {
        "K": 0,
        "M": 0,
        "U": 0,
        # "irrelevant": len(remove_uid),
        "irrelevant": 0,
    }

    for u, v in users.items():
        if v["Junk"] > 0:
            continue
        if v["proM"] > v["proK"]:
            cnt["M"] += 1
        elif v["proM"] < v["proK"]:
            cnt["K"] += 1
        elif v["proM"] > 0 or v["proK"] > 0:
            cnt["U"] += 1
        else:
            cnt["irrelevant"] += 1

    print(dt, cnt)

    if not bots:
        sess.add(Weekly_Predict(dt=dt,
                                U_Cristina=cnt["K"],
                                U_Macri=cnt["M"],
                                U_unclassified=cnt["U"],
                                U_irrelevant=cnt["irrelevant"]))
    else:
        sess.add(Bot_Weekly_Predict(dt=dt,
                                    U_Cristina=cnt["K"],
                                    U_Macri=cnt["M"],
                                    U_unclassified=cnt["U"],
                                    U_irrelevant=cnt["irrelevant"]))

    sess.commit()


def predict_cumulative(dt, prob=0.68, clear=False):
    """
    """
    sess = get_session()
    if clear:
        sess.query(History_Predict).filter(History_Predict.dt == dt).delete()
        sess.commit()

    # save
    if not os.path.exists(f"disk/cul_from_March_1/{dt.to_date_string()}-{prob}.txt"):
        predict_cumulative_file(dt, dt, prob=prob)

    print("load ~", f"disk/cul_from_March_1/{dt.to_date_string()}-{prob}.txt")
    cul_today = json.load(open(f"disk/cul_from_March_1/{dt.to_date_string()}-{prob}.txt"))

    cnt = {
        "dt": dt.to_date_string(),
        "K": 0,
        "M": 0,
        "U": 0,
        "I": 0,
    }

    # user-level
    for u, v in cul_today.items():
        if v["I"] > 0:
            continue
        if v["M"] > v["K"]:
            cnt["M"] += 1
        elif v["M"] < v["K"]:
            cnt["K"] += 1
        elif v["M"] > 0 or v["K"] > 0:
            cnt["U"] += 1
        else:
            cnt["I"] += 1
            
    print("% of K", cnt["K"] / (cnt["K"] + cnt["M"] + cnt["U"] + cnt["I"]))

    sess.add(History_Predict(
        dt=dt,
        U_Cristina=cnt["K"],
        U_Macri=cnt["M"],
        U_unclassified=cnt["U"],
        U_irrelevant=cnt["I"])
    )
    sess.commit()
    sess.close()


def predict_cumulative_to_csv(start, end, in_dir, prob=0.68):
    """
    从历史开始每天累积，存储到csv
    """
    rsts = []
    for dt in pendulum.period(start, end):
        print("load ~", f"disk/cul_{in_dir}/{dt.to_date_string()}-{prob}.txt")
        cul_today = json.load(open(f"disk/cul_{in_dir}/{dt.to_date_string()}-{prob}.txt"))
        cnt = get_camp_count_from_users(cul_today)
        cnt["dt"] = dt.to_date_string()
        print(cnt)
        rsts.append(cnt)
    pd.DataFrame(rsts).set_index("dt").to_csv(f"data/cul_start_{in_dir}_{prob}.csv")
            
    
def predict_dir_to_csv(start, end, in_dir, prob=0.68):
    """
    从14days数据，存储到csv
    """
    rsts = []
    for dt in pendulum.period(start, end):
        print("load ~", f"disk/{in_dir}/{dt.to_date_string()}-{prob}.txt")
        cul_today = json.load(open(f"disk/{in_dir}/{dt.to_date_string()}-{prob}.txt"))
        cnt = get_camp_count_from_users(cul_today)
        cnt["dt"] = dt.to_date_string()
        print(cnt)
        rsts.append(cnt)
    pd.DataFrame(rsts).set_index("dt").to_csv(f"data/{in_dir}_{prob}.csv")


def predict_culmulative_swing_loyal(start, end, prob=0.68):
    """
    从历史开始每天累积
    """
    rsts = []
    intention = {}
    
    _period = pendulum.Period(start, end)
    for dt in _period:
        print(dt)

        # save
        print("load ~", f"disk/cul_from_March_1/{dt.to_date_string()}-{prob}.txt")
        cul_today = json.load(open(f"disk/cul_from_March_1/{dt.to_date_string()}-{prob}.txt"))

        # user-level
        for u, v in cul_today.items():
            if u not in intention:
                intention[u] = "new"
                if v["I"] > 0:
                    intention[u] = "JUNK"
                elif v["M"] > v["K"]:
                    intention[u] = "loyal MP"
                elif v["M"] < v["K"]:
                    intention[u] = "loyal FF"
                elif v["M"] == 0 and v["K"] == 0:
                    intention[u] = "loyal Others"
                else:
                    intention[u] = "swing Others"
        
            else:
                _int = intention[u]
                if v["I"] > 0:
                    intention[u] = "JUNK"
                    
                if _int == "loyal MP":
                    if v["M"] > v["K"]:
                        continue
                    elif v["M"] == v["K"]:
                        intention[u] = "swing Others"
                    elif v["M"] < v["K"]:
                        intention[u] = "swing FF"
                        
                elif _int == "loyal FF":
                    if v["K"] > v["M"]:
                        continue
                    elif v["M"] == v["K"]:
                        intention[u] = "swing Others"
                    elif v["K"] < v["M"]:
                        intention[u] = "swing MP"
                        
                elif _int == "swing MP":
                    if v["M"] > 2 * v["K"]:
                        intention[u] = "loyal MP"
                    elif v["M"] > v["K"]:
                        continue
                    elif v["M"] == v["K"]:
                        intention[u] = "swing Others"
                    elif v["M"] < v["K"]:
                        intention[u] = "swing FF"
                                 
                elif _int == "swing FF":
                    if v["K"] > 2 * v["M"]:
                        intention[u] = "loyal FF"
                    elif v["K"] > v["M"]:
                        continue
                    elif v["M"] == v["K"]:
                        intention[u] = "swing Others"
                    elif v["K"] < v["M"]:
                        intention[u] = "swing MP"
                        
                elif _int == "loyal Others":
                    if v["K"] == ["M"] and v["K"] > 0:
                        intention[u] = "swing Others"
                    elif v["K"] > v["M"]:
                        intention[u] = "loyal FF"
                    elif v["K"] < v["M"]:
                        intention[u] = "loyal MP"
  
                elif _int == "swing Others":
                    if v["K"] > v["M"]:
                        intention[u] = "swing FF"
                    elif v["K"] < v["M"]:
                        intention[u] = "swing MP"
              
        cnt = {
            "dt": dt.to_date_string(),
            "loyal FF": 0,
            "swing FF": 0,
            "swing Others": 0,
            "swing MP": 0,
            "loyal MP": 0,
            "loyal Others": 0,
            "JUNK": 0
        }
                                                             
        for u, v in intention.items():
            cnt[v] += 1
        print(cnt)
        
        rsts.append(cnt)
    
    pd.DataFrame(rsts).set_index("dt").to_csv(f"data/swings_and_loyals-end-{dt.to_date_string()}.csv")



def predict_culmulative_swing_loyal_v2(start, end, prob=0.68):
    """
    从历史开始每天累积
    """
    rsts = []
    intention = {}
    
    _period = pendulum.Period(start, end)
    for dt in _period:
        print(dt)

        # save
        print("load ~", f"disk/cul_from_March_1/{dt.to_date_string()}-{prob}.txt")
        cul_today = json.load(open(f"disk/cul_from_March_1/{dt.to_date_string()}-{prob}.txt"))

        # user-level
        for u, v in cul_today.items():
            if u not in intention:
                intention[u] = "new"
                if v["I"] > 0:
                    intention[u] = "JUNK"
                elif v["M"] > v["K"]:
                    intention[u] = "loyal MP"
                elif v["M"] < v["K"]:
                    intention[u] = "loyal FF"
                elif v["M"] == 0 and v["K"] == 0:
                    intention[u] = "loyal Others"
                else:
                    intention[u] = "swing Others"
        
            else:
                _int = intention[u]
                if v["I"] > 0:
                    intention[u] = "JUNK"
                    
                if _int == "loyal MP":
                    if v["M"] > v["K"]:
                        continue
                    elif v["M"] == v["K"]:
                        intention[u] = "swing Others"
                    elif v["M"] < v["K"]:
                        intention[u] = "swing FF"
                        
                elif _int == "loyal FF":
                    if v["K"] > v["M"]:
                        continue
                    elif v["M"] == v["K"]:
                        intention[u] = "swing Others"
                    elif v["K"] < v["M"]:
                        intention[u] = "swing MP"
                        
                elif _int == "swing MP":
                    if v["M"] > 2 * v["K"]:
                        intention[u] = "loyal MP"
                    elif v["M"] > v["K"]:
                        continue
                    elif v["M"] == v["K"]:
                        intention[u] = "swing Others"
                    elif v["M"] < v["K"]:
                        intention[u] = "swing FF"
                                 
                elif _int == "swing FF":
                    if v["K"] > 2 * v["M"]:
                        intention[u] = "loyal FF"
                    elif v["K"] > v["M"]:
                        continue
                    elif v["M"] == v["K"]:
                        intention[u] = "swing Others"
                    elif v["K"] < v["M"]:
                        intention[u] = "swing MP"
                        
                elif _int == "loyal Others":
                    if v["K"] == ["M"] and v["K"] > 0:
                        intention[u] = "swing Others"
                    elif v["K"] > v["M"]:
                        intention[u] = "loyal FF"
                    elif v["K"] < v["M"]:
                        intention[u] = "loyal MP"
  
                elif _int == "swing Others":
                    if v["K"] > v["M"]:
                        intention[u] = "swing FF"
                    elif v["K"] < v["M"]:
                        intention[u] = "swing MP"
              
        cnt = {
            "dt": dt.to_date_string(),
            "loyal FF": 0,
            "swing FF": 0,
            "swing Others": 0,
            "swing MP": 0,
            "loyal MP": 0,
            "loyal Others": 0,
            "JUNK": 0
        }
                                                             
        for u, v in intention.items():
            cnt[v] += 1
        print(cnt)
        
        rsts.append(cnt)
    
    pd.DataFrame(rsts).set_index("dt").to_csv(f"data/swings_and_loyals-end-{dt.to_date_string()}.csv")
       

def predict_culmulative_user_class(start, end, prob=0.68):
    """
    从历史开始每天累积
    """
    _period = pendulum.Period(start, end)
    for dt in _period:
        print(dt)
        # save
        print("load ~", f"disk/cul_from_March_1/{dt.to_date_string()}-{prob}.txt")
        cul_today = json.load(open(f"disk/cul_from_March_1/{dt.to_date_string()}-{prob}.txt"))

        # cnt = {
        #     "dt": dt.to_date_string(),
        #     "Fl": [],
        #     "Ml": [],
        #     "Fs": [],
        #     "Ms": [],
        #     "U": [],
        #     "I": [],
        # }
        # # user-level
        # for u, v in cul_today.items():
        #     if v["I"] > 0:
        #         continue
        #     if v["M"] > v["K"]:
        #         if v["M"] > (v["K"] * 2):
        #             cnt["Ml"].append(u)
        #         else:
        #             cnt["Ms"].append(u)
        #     elif v["M"] < v["K"]:
        #         if v["K"] > (v["M"] * 2):
        #             cnt["Fl"].append(u)
        #         else:
        #             cnt["Fs"].append(u)
        #     elif v["M"] > 0 or v["K"] > 0:
        #         cnt["U"].append(u)
        #     else:
        #         cnt["I"].append(u)
    
        # print("save ~", f"disk/user_class/{dt.to_date_string()}-{prob}.txt")
        # json.dump(cnt, open(f"disk/user_class/{dt.to_date_string()}-{prob}.txt", "w"))
    

def new_users_in_different_class(start, end, w=14, prob=0.68):
    rsts = []
    for dt in pendulum.Period(start, end):
        print("load ~", f"disk/cul_from_March_1/{dt.to_date_string()}-{prob}.txt")
        previous = json.load(open(f"disk/cul_from_March_1/{dt.to_date_string()}-{prob}.txt"))
        previous_users = set(previous.keys())
        today_str = dt.add(days=w).to_date_string()
        if os.path.exists(f'disk/cul_from_March_1/{today_str}-{prob}.txt'):
            today = json.load(open(f'disk/cul_from_March_1/{today_str}-{prob}.txt'))
        else:
            break
        today_users = set(today.keys())            
        
        new_users = today_users - previous_users
        cnt = {
            "dt": today_str,
            "New users (FF)": 0,
            "New users (MP)": 0,
            "New users (Others)": 0
        }
        for u in new_users:
            if today[u]["K"] > today[u]["M"]:
                cnt["New users (FF)"] += 1
            elif today[u]["K"] < today[u]["M"]:
                cnt["New users (MP)"] += 1
            else:
                cnt["New users (Others)"] += 1
        print(cnt)
        rsts.append(cnt)
    pd.DataFrame(rsts).set_index("dt").to_csv(f"data/new-users-end-{today_str}-{w}.csv")
    
    
def old_users_in_different_class(start, end, peri=1, w=14, prob=0.68, norm=False):
    import ujson as json
    
    rsts = []            
    for dt in pendulum.Period(start, end):
        today_str = dt.add(days=peri * w).to_date_string()
        if not os.path.exists(f'disk/users-14days/{today_str}-{prob}.txt'):
            break
        
        live_K = []
        live_M = []
        live_U = []
        
        always_K_users = None
        always_M_users = None            
        always_U_users = None
                    
        for i in range(peri):
            print(f"period={peri}, i={i}")
            print("load ~", f"disk/users-14days/{dt.add(days=i * w + w).to_date_string()}-{prob}.txt")
            prev = json.load(open(f"disk/users-14days/{dt.add(days=i * w + w).to_date_string()}-{prob}.txt"))
            K_users = set()
            M_users = set()
            U_users = set()
            for u, v in prev.items():
                if v["I"] > 0:
                    continue
                if v["K"] > v["M"]:
                    K_users.add(u)
                elif v["K"] < v["M"]:
                    M_users.add(u)
                else:
                    U_users.add(u)
            
            live_K.append(K_users)
            live_M.append(M_users)
            live_U.append(U_users)
            print(len(live_K[i]), len(live_M[i]), len(live_U[i]))


        for i in range(peri):
            Ku = live_K[i]
            # print("Users of K:", len(Ku))
            if always_K_users is None:
                always_K_users = Ku
            else:
                always_K_users = always_K_users & Ku
            # print("Union:", len(always_K_users))

        for i in range(peri):
            Mu = live_M[i]
            # print("Users of M:", len(Mu))
            if always_M_users is None:
                always_M_users = Mu
            else:
                always_M_users = always_M_users & Mu
            print("Union:", len(always_M_users))            
                
        for i in range(peri):
            Uu = live_U[i]
            # print("Users of U:", len(Uu))
            if always_U_users is None:
                always_U_users = Uu
            else:
                always_U_users = always_U_users & Uu
            # print("Union:", len(always_U_users))

        cnt = {
            "dt": today_str,
            "users (FF)": len(always_K_users),
            "users (MP)": len(always_M_users),
            "users (Others)": len(always_U_users)            
        }

        # if norm:
        #     today = json.load(open(f'disk/cul_from_March_1/{today_str}-{prob}.txt'))
        #     today_users_set = get_user_set(today)
        #     cnt["users (FF)"] /= len(today_users_set["K"])
        #     cnt["users (MP)"] /= len(today_users_set["M"])
        #     cnt["users (Other)"] /= len(today_users_set["U"])

        print(cnt)
        rsts.append(cnt)
    
    if norm:
        pd.DataFrame(rsts).set_index("dt").to_csv(f"data/permenant-users-end-{today_str}-{peri}-norm.csv")
    else:
        pd.DataFrame(rsts).set_index("dt").to_csv(f"data/permenant-users-end-{today_str}-{peri}.csv")
    

def predict_cumulative_file(start, end, prob=0.68, out_dir="from_March_1_v2"):
    """
    从历史开始每天累积，累计用户的tweets
    """
    # users = {}
    def union_users(u1, u2):
        u_temp = {}
        for uid, v in u1.items():
            u_temp[uid] = {}
            u_temp[uid]["M"] = v["M"]
            u_temp[uid]["K"] = v["K"]
            u_temp[uid]["U"] = v["U"]
            u_temp[uid]["I"] = v["I"]
        for uid, v in u2.items():
            if uid not in u_temp:
                u_temp[uid] = {
                    "M": 0,
                    "K": 0,
                    "U": 0,
                    "I": 0,
                }
            u_temp[uid]["M"] += v["M"]
            u_temp[uid]["K"] += v["K"]
            u_temp[uid]["U"] += v["U"]
            u_temp[uid]["I"] += v["I"]
        return u_temp

    for dt in pendulum.Period(start, end):
        # print(dt)
        if dt.to_date_string() <= "2019-03-02":
            continue
        # 在预测的时间序列上，永远是不包含end！也就是说3月2日的预测，实际用的是3月1日的数据
        yesterday_str = dt.add(days=-1).to_date_string()
        out_name = f"disk/cul_{out_dir}/{dt.to_date_string()}-{prob}.txt"
        if Path(out_name).exists():
            continue
        cul_yesterday = json.load(open(f"disk/cul_{out_dir}/{yesterday_str}-{prob}.txt"))
        users_today = json.load(open(f"disk/users_v2/{yesterday_str}-{prob}.txt"))
        cul_today = union_users(users_today, cul_yesterday)
        # save
        print("save ~", out_name)
        json.dump(cul_today, open(f"disk/cul_{out_dir}/{dt.to_date_string()}-{prob}.txt", "w"))
    

def predict_cumulative_file_ignore(start, end, prob=0.68):
    """
    从历史开始每天累积
    """
    # users = {}
    def union_users(u1, u2):
        u_temp = {}
        for uid, v in u1.items():
            u_temp[uid] = {}
            u_temp[uid]["M"] = v["M"]
            u_temp[uid]["K"] = v["K"]
            u_temp[uid]["U"] = v["U"]
            u_temp[uid]["I"] = v["I"]
        for uid, v in u2.items():
            if uid not in u_temp:
                u_temp[uid] = {
                    "M": 0,
                    "K": 0,
                    "U": 0,
                    "I": 0,
                }
            u_temp[uid]["M"] += v["M"]
            u_temp[uid]["K"] += v["K"]
            u_temp[uid]["U"] += v["U"]
            u_temp[uid]["I"] += v["I"]
        return u_temp

    for dt in pendulum.Period(start, end):
        print(dt)
        if dt.to_date_string() <= "2019-03-01":
            continue
        # 在预测的时间序列上，永远是不包含end！也就是说3月2日的预测，实际用的是3月1日的数据
        yesterday_str = dt.add(days=-1).to_date_string()
        cul_yesterday = json.load(
            open(f"disk/cul_from_March_1_ignore1/{yesterday_str}-{prob}.txt"))
        users_today = json.load(open(f"disk/users/{yesterday_str}-{prob}.txt"))
        cul_today = union_users(users_today, cul_yesterday)
        # save
        print(
            "save ~", f"disk/cul_from_March_1/{dt.to_date_string()}-{prob}.txt")
        json.dump(cul_today, open(
            f"disk/cul_from_March_1/{dt.to_date_string()}-{prob}.txt", "w"))
        
        
def save_today_user_snapshot(sess, now, prob):
    """
    保存每天用户的行为快照，在每天的数据下，应该是实际当天的日期。
    """
    tweets = get_tweets(sess, now, now.add(days=1))

    users = {}
    for t in tqdm(tweets):
        uid = t.user_id
        if uid not in users:
            users[uid] = {
                "M": 0,
                "K": 0,
                "U": 0,
                "I": 0,
            }

        if t.proM < 0:
            users[uid]["I"] += 1
        elif t.proM >= prob:
            users[uid]["M"] += 1
        elif t.proM < 1 - prob:
            users[uid]["K"] += 1
        else:
            users[uid]["U"] += 1

    json.dump(users, open(
        f"disk/users_v2/{now.to_date_string()}-{prob}.txt", "w"))


def save_today_user_snapshot_ignore(sess, now, prob, ignore_id):
    """
    保存每天用户的行为快照，在每天的数据下，应该是实际当天的日期。
    """
    tweets = get_tweets(sess, now, now.add(days=1))
    users = {}
    for t in tqdm(tweets):
        if t.tweet_id in ignore_id:
            continue
        uid = t.user_id
        if uid not in users:
            users[uid] = {
                "M": 0,
                "K": 0,
                "U": 0,
                "I": 0,
            }

        if t.proM < 0:
            users[uid]["I"] += 1
        elif t.proM >= prob:
            users[uid]["M"] += 1
        elif t.proM < 1 - prob:
            users[uid]["K"] += 1
        else:
            users[uid]["U"] += 1

    json.dump(users, open(
        f"disk/users-ignore1/{now.to_date_string()}-{prob}.txt", "w"))
    
    
def save_user_snapshot(start, end, w=14, prob=0.66):
    def union_users(u1, u2):
        u_temp = {}
        for uid, v in u1.items():
            u_temp[uid] = {}
            u_temp[uid]["M"] = v["M"]
            u_temp[uid]["K"] = v["K"]
            u_temp[uid]["U"] = v["U"]
            u_temp[uid]["I"] = v["I"]
        for uid, v in u2.items():
            if uid not in u_temp:
                u_temp[uid] = {
                    "M": 0,
                    "K": 0,
                    "U": 0,
                    "I": 0,
                }
            u_temp[uid]["M"] += v["M"]
            u_temp[uid]["K"] += v["K"]
            u_temp[uid]["U"] += v["U"]
            u_temp[uid]["I"] += v["I"]
        return u_temp
    
    for dt in pendulum.Period(start, end):
        users = {}
        print(dt.to_date_string())
        _end = dt.add(days=w-1)
        for now in pendulum.Period(dt, _end):
            _u = json.load(open(f"F:\ARG2019\daily-csv\{now.to_date_string()}-{prob}.txt"))
            users = union_users(users, _u)

        json.dump(users, open(
            f"disk/users-{w}days/{_end.to_date_string()}-{prob}.txt", "w"))
    

def new_save_user_snapshot(start, end, w=14, prob=0.66):
    
    def read_tweets(in_name):
        _users = {}
        for line in open(in_name):
            w = line.strip().split(',')
            u = {
                "M": 0,
                "K": 0,
                "U": 0,
                "I": 0,
            }
            p = float(w[2])
            if p > prob:
                u["M"] += 1
            elif p > (1 - prob):
                u["U"] += 1
            elif p > 0:
                u["K"] += 1
            else:
                u["I"] += 1
            _users[w[1]] = u
        return _users
            
    def union_users(u1, u2):
        u_temp = {}
        for uid, v in u1.items():
            u_temp[uid] = {}
            u_temp[uid]["M"] = v["M"]
            u_temp[uid]["K"] = v["K"]
            u_temp[uid]["U"] = v["U"]
            u_temp[uid]["I"] = v["I"]
        for uid, v in u2.items():
            if uid not in u_temp:
                u_temp[uid] = {
                    "M": 0,
                    "K": 0,
                    "U": 0,
                    "I": 0,
                }
            u_temp[uid]["M"] += v["M"]
            u_temp[uid]["K"] += v["K"]
            u_temp[uid]["U"] += v["U"]
            u_temp[uid]["I"] += v["I"]
        return u_temp
    
    for dt in pendulum.Period(start, end):
        users = {}
        print(dt.to_date_string())
        _end = dt.add(days=w-1)
        for now in pendulum.Period(dt, _end):
            try:
                _u = read_tweets(f"F:/ARG2019/daily-csv/{now.to_date_string()}.csv")
                users = union_users(users, _u)
            except:
                print("ERROR FILE:", now.to_date_string())
                
        json.dump(users, open(
            f"disk/users-{w}days/{_end.to_date_string()}-{prob}.txt", "w"))

    
def get_camp_count_from_users(_users):
    cnt = {
        "FF": 0,
        "MP": 0,
        "Others": 0,
    }
    for u, v in _users.items():
        if v["I"] > 0:
            continue
        if v["K"] > v["M"]:
            cnt["FF"] += 1
        elif v["M"] > v["K"]:
            cnt["MP"] += 1
        else:
            cnt["Others"] += 1
    return cnt

    
def predict_user_snapshot(win=7):
    """
    7天为时间窗口的用户快照
    """
    # keep_set = set([int(line.strip()) for line in open("data/0731-week-keep.txt")])
    # Cristina_set = set([int(line.strip()) for line in open("data/0731-week-Cristna.txt")])
    # elecciones_set = set([int(line.strip()) for line in open("data/0731-week-elecciones.txt")])

    # keep_set = set([int(line.strip()) for line in open("data/0731-week-terms1.txt")])
    # keep_set = set([int(line.strip()) for line in open("data/0731-week-terms2.txt")])
    sess = get_session()

    start = pendulum.datetime(2019, 3, 1, tz="UTC")
    end = pendulum.datetime(2019, 8, 21, tz="UTC")
    _period = pendulum.Period(start, end)

    for dt in _period:
        print(dt)
        users = {}
        tweets = get_tweets(sess, dt.add(days=-win), dt)

        for t in tqdm(tweets):
            uid = t.user_id
            if uid not in users:
                users[uid] = {
                    "M": 0,
                    "K": 0,
                    "U": 0,
                    "I": 0,
                }
            if t.proM < 0:
                users[uid]["I"] += 1
            elif t.proM >= 0.68:
                users[uid]["M"] += 1
            elif t.proM < 0.32:
                users[uid]["K"] += 1
            else:
                users[uid]["U"] += 1

        json.dump(users, open(
            f"disk/users/{dt.to_date_string()}-0.68.txt", "w"))
        
        # if win == 7:
        #     json.dump(users, open(f"disk/users/{dt.to_date_string()}.txt", "w"))
        # else:
        #     json.dump(users, open(f"disk/users-{win}days/{dt.to_date_string()}.txt", "w"))

        # cnt = {
        #     "K": 0,
        #     "M": 0,
        #     "U": 0,
        #     "I": 0,
        # }

        # for u, v in users.items():
        #     if v["M"] > v["K"]:
        #         cnt["M"] += 1
        #     elif v["M"] < v["K"]:
        #         cnt["K"] += 1
        #     elif v["M"] > 0 or v["K"] > 0:
        #         cnt["U"] += 1
        #     else:
        #         cnt["I"] += 1

        # print(dt, cnt)

    sess.close()


def predict_user_before_PASO(p):
    """
    PASO分析，设置不同的t_0
    """
    start = pendulum.datetime(2019, 3, 1, tz="UTC")
    end = pendulum.datetime(2019, 8, 10, tz="UTC")
    dt_users = {}
    # load user-data
    for dt in pendulum.Period(start, end):
        dt = dt.to_date_string()
        print("loading ...", dt)
        users = json.load(open(f"disk/users/{dt}-{p}.txt"))
        dt_users[dt] = users

    def union_users(u1, u2):
        u_temp = {}
        for uid, v in u1.items():
            u_temp[uid] = {}
            u_temp[uid]["M"] = v["M"]
            u_temp[uid]["K"] = v["K"]
            u_temp[uid]["U"] = v["U"]
            u_temp[uid]["I"] = v["I"]
        for uid, v in u2.items():
            if uid not in u_temp:
                u_temp[uid] = {
                    "M": 0,
                    "K": 0,
                    "U": 0,
                    "I": 0,
                }
            u_temp[uid]["M"] += v["M"]
            u_temp[uid]["K"] += v["K"]
            u_temp[uid]["U"] += v["U"]
            u_temp[uid]["I"] += v["I"]
        return u_temp

    start = pendulum.datetime(2019, 3, 1, tz="UTC")
    end = pendulum.datetime(2019, 8, 10, tz="UTC")
    now = end
    cul_users = {}
    while now >= start:
        print(now)
        now_str = now.to_date_string()
        cul_users = union_users(cul_users, dt_users[now_str])
        json.dump(cul_users, open(f"disk/PASO/{now_str}-{p}.txt", "w"))
        now = now.add(days=-1) # 日期不断前推


def predict_percent(sess, dt, clear=False):
    """
    use tweets in the last 14 (lag) days to predict everyday
    dt is today,
    so, start is -14 day, end is -1 day.
    save -1 day in the db
    """

    if clear:
        sess.query(Percent).filter(Percent.dt == dt).delete()
        sess.commit()

    # 2019-06-27 updates
    # k = (0.67, 0.27, 0.525)
    k = (1, 0)
    r = sess.query(Weekly_Predict).filter(Weekly_Predict.dt == dt).one()

    M_pro = r.U_Macri / (r.U_Cristina + r.U_Macri)
    M_pro = M_pro * k[0] + k[1]
    K_pro = 1 - M_pro

    print(dt, K_pro, M_pro)
    sess.add(Percent(dt=dt, K=K_pro, M=M_pro))
    sess.commit()


def get_percent(sess, dt, clas=2):
    dt = pendulum.parse(dt)
    r = sess.query(Percent).filter(Percent.dt == dt).one()
    K_pro = r.K
    M_pro = r.M

    if clas == 2:
        # print(dt, r.K, r.M)
        return r.K, r.M

    elif clas == 3:
        r = sess.query(Weekly_Predict).filter(Weekly_Predict.dt == dt).one()
        U_pro = r.U_unclassified / \
            (r.U_Cristina + r.U_Macri + r.U_unclassified)
        left = 1 - U_pro
        K_pro3 = left * K_pro
        M_pro3 = left * M_pro
        # print(dt, K_pro3, M_pro3, U_pro)
        return K_pro3, M_pro3, U_pro


def predict_Ndays(start, end, win=7, clear=False):
    sess = get_session()
    if win == 3:
        Target = Day3_Predict
    elif win == 7:
        Target = Day7_Predict
    elif win == 14:
        Target = Day14_Predict
    elif win == 30:
        Target = Day30_Predict
    elif win == 60:
        Target = Day60_Predict

    if clear:
        sess.query(Target).filter(Target.dt >= start, Target.dt <= end).delete()
        sess.commit()

    for dt in pendulum.Period(start, end):
        users = {}
        # print("predict daily!", start, "~", end)
        tweets = get_tweets(sess, dt.add(days=-win), dt)
        # remove_uid = set()

        for t in tweets:
            uid = t.user_id
            # if uid in remove_uid:
            #     continue
            if uid not in users:
                users[uid] = {
                    "proM": 0,
                    "proK": 0,
                    "Unclassified": 0,
                }

            if t.proM < 0:
                # remove_uid.add(uid)
                # if uid in users:
                #     users.pop(uid)
                continue
            elif t.proM > 0.68:
                users[uid]["proM"] += 1
            elif t.proM < 0.32:
                users[uid]["proK"] += 1
            else:
                users[uid]["Unclassified"] += 1

        cnt = {
            "K": 0,
            "M": 0,
            "U": 0,
            # "irrelevant": len(remove_uid),
            "irrelevant": 0,
        }

        for u, v in users.items():
            if v["proM"] > v["proK"]:
                cnt["M"] += 1
            elif v["proM"] < v["proK"]:
                cnt["K"] += 1
            elif v["proM"] > 0 or v["proK"] > 0:
                cnt["U"] += 1
            else:
                cnt["irrelevant"] += 1

        print(dt, cnt)
        sess.add(Target(dt=dt,
                        U_Cristina=cnt["K"],
                        U_Macri=cnt["M"],
                        U_unclassified=cnt["U"],
                        U_irrelevant=cnt["irrelevant"]))

        sess.commit()
        sess.close()


# run it each day
def predict3_day(sess, dt, lag=14, clear=False):
    """
    use tweets in the last 14 (lag) days to predict everyday
    dt is today,
    so, start is -14 day, end is -1 day.
    save -1 day in the db
    """
    if clear:
        sess.query(Weekly_Predict3).filter(Weekly_Predict3.dt == dt).delete()
        sess.commit()
    start = dt.add(days=-lag)
    end = dt

    users = {}
    print("predict3 daily!", start, "~", end)
    tweets = get_tweets(sess, start, end)

    massa_tweets = set([t[0] for t in get_tweets3(sess, start, end)])

    remove_uid = set()

    for t in tqdm(tweets):
        uid = t.user_id
        if uid in remove_uid:
            continue
        if uid not in users:
            users[uid] = {
                "proM": 0,
                "proK": 0,
                "proA": 0,
                "Unclassified": 0,
            }

        if t.proM < 0:
            remove_uid.add(uid)
            if uid in users:
                users.pop(uid)
        elif t.tweet_id in massa_tweets:
            users[uid]["proA"] += 1
        elif t.proM > 0.75:
            users[uid]["proM"] += 1
        elif t.proM < 0.25:
            users[uid]["proK"] += 1
        else:
            users[uid]["Unclassified"] += 1

    cnt = {
        "K": 0,
        "M": 0,
        "A": 0,
        "U": 0,
        "irrelevant": len(remove_uid),
    }

    for u, v in users.items():
        if v["proA"] > v["proM"] and v["proA"] > v["proK"]:
            cnt["A"] += 1
        elif v["proM"] > v["proK"]:
            cnt["M"] += 1
        elif v["proM"] < v["proK"]:
            cnt["K"] += 1
        elif v["proM"] > 0 or v["proK"] > 0:
            cnt["U"] += 1
        else:
            cnt["irrelevant"] += 1

    sess.add(Weekly_Predict3(dt=dt,
                             U_Cristina=cnt["K"],
                             U_Macri=cnt["M"],
                             U_Massa=cnt["A"],
                             U_unclassified=cnt["U"],
                             U_irrelevant=cnt["irrelevant"]))

    sess.commit()


def predict_day_paso(paso_tids, sess, dt, lag=14):
    """
    use tweets in the last 14 (lag) days to predict everyday
    dt is today,
    so, start is -14 day, end is -1 day.
    save -1 day in the db
    """
    start = dt.add(days=-lag)
    end = dt
    users = {}
    tweets = get_tweets(sess, start, end, bots=False)
    remove_uid = set()

    for t in tweets:
        if t.tweet_id in paso_tids:
            continue

        uid = t.user_id
        if uid in remove_uid:
            continue
        if uid not in users:
            users[uid] = {
                "proM": 0,
                "proK": 0,
                "Unclassified": 0,
            }

        if t.proM < 0:
            remove_uid.add(uid)
            if uid in users:
                users.pop(uid)
        elif t.proM > 0.75:
            users[uid]["proM"] += 1
        elif t.proM < 0.25:
            users[uid]["proK"] += 1
        else:
            users[uid]["Unclassified"] += 1

    cnt = {
        "K": 0,
        "M": 0,
        "U": 0,
        "irrelevant": len(remove_uid),
    }

    for u, v in users.items():
        if v["proM"] > v["proK"]:
            cnt["M"] += 1
        elif v["proM"] < v["proK"]:
            cnt["K"] += 1
        elif v["proM"] > 0 or v["proK"] > 0:
            cnt["U"] += 1
        else:
            cnt["irrelevant"] += 1

    print(dt, cnt)
    sess.add(NoPASO_Predict(dt=dt,
                            U_Cristina=cnt["K"],
                            U_Macri=cnt["M"],
                            U_unclassified=cnt["U"],
                            U_irrelevant=cnt["irrelevant"]))
    sess.commit()


def terms_stat():
    """
    terms中分类统计
    """
    sess = get_session()

    def _get_tweets_json():
        target_dir = ["201902", "201903", "201904", "201905"]
        for _dir in target_dir:
            for in_name in tqdm(os.listdir("disk/" + _dir)):
                if in_name.endswith("PRO.txt") or in_name.endswith("Moreno.txt") or in_name.endswith("Sola.txt"):
                    continue
                return_name = in_name[7:-4]
                in_name = "disk/" + _dir + "/" + in_name
                print(return_name)
                for line in open(in_name):
                    d = json.loads(line.strip())
                    yield return_name, d["id"]

    def _get_proM(tid):
        try:
            _m = sess.query(Tweet.proM).filter(Tweet.tweet_id == tid).one()[0]
        except:
            _m = -1
        return _m

    dict_terms = {}
    tweets_json = _get_tweets_json()
    for name, tweet_id in tweets_json:
        _proM = _get_proM(tweet_id)
        if _proM == -1:
            print(tweet_id, "not exists.")
            continue
        # print(_proM)
        if name not in dict_terms:
            dict_terms[name] = [0, 0, 0]  # macri, cristina, un
        if _proM > 0.75:
            dict_terms[name][0] += 1
        elif _proM < 0.25:
            dict_terms[name][1] += 1
        else:
            dict_terms[name][2] += 1

    terms_data = []
    for name, v in dict_terms.items():
        terms_data.append(
            Term(name=name, proM=v[0], proK=v[1], unclassified=v[2])
        )

    sess.query(Term).delete()
    sess.commit()

    sess.add_all(terms_data)
    sess.commit()


def terms_month_stat():
    """
    terms中分类统计
    """
    sess = get_session()

    def _get_tweets_json():
        target_dir = ["201905"]
        for _dir in target_dir:
            for in_name in tqdm(os.listdir("disk/" + _dir)):
                if in_name.endswith("PRO.txt") or in_name.endswith("Moreno.txt") or in_name.endswith("Sola.txt"):
                    continue
                return_name = in_name[7:-4]
                in_name = "disk/" + _dir + "/" + in_name
                print(return_name)
                for line in open(in_name):
                    d = json.loads(line.strip())
                    yield return_name, d["id"]

    def _get_proM(tid):
        try:
            _m = sess.query(Tweet.proM).filter(Tweet.tweet_id == tid).one()[0]
        except:
            _m = -1
        return _m

    dict_terms = {}
    tweets_json = _get_tweets_json()
    for name, tweet_id in tweets_json:
        _proM = _get_proM(tweet_id)
        if _proM == -1:
            print(tweet_id, "not exists.")
            continue
        # print(_proM)
        if name not in dict_terms:
            dict_terms[name] = [0, 0, 0]  # macri, cristina, un
        if _proM > 0.75:
            dict_terms[name][0] += 1
        elif _proM < 0.25:
            dict_terms[name][1] += 1
        else:
            dict_terms[name][2] += 1

    terms_data = []
    for name, v in dict_terms.items():
        terms_data.append(
            Month_Term(name=name, proM=v[0], proK=v[1], unclassified=v[2])
        )

    sess.query(Month_Term).delete()
    sess.commit()

    sess.add_all(terms_data)
    sess.commit()


def tweets_to_new_clas(sess):
    """
    导入全部数据，并分类
    """
    tweets_json = get_tweets_json()
    for d, tweet_id in tweets_json:
        tweets_data.append(
            New_clas(tweet_id=tweet_id))

        words = bag_of_words_and_bigrams(tokenizer.tokenize(d["text"]))
        X.append(words)

        if len(tweets_data) == 5000:
            # y = clf1.predict_proba(v1.transform(X))
            # for i in range(len(y)):
            #     # print(y)
            #     # return -1
            #     proK = round(y[i][0], 4)
            #     proM = round(y[i][1], 4)
            #     proA = round(y[i][2], 4)
            #     tweets_data[i].proK3 = proK
            #     tweets_data[i].proM3 = proM
            #     tweets_data[i].proA3 = proA
                # print(y[i])

            y = clf2.predict_proba(v2.transform(X))
            for i in range(len(y)):
                # print(y.shape)
                proK = round(y[i][0], 4)
                proM = round(y[i][1], 4)
                tweets_data[i].proK = proK
                tweets_data[i].proM = proM

            sess.add_all(tweets_data)
            sess.commit()
            X = []
            tweets_data = []

    if tweets_data:
        # y = clf1.predict_proba(v1.transform(X))
        # for i in range(len(y)):
        #     proK = round(y[i][0], 4)
        #     proM = round(y[i][1], 4)
        #     proA = round(y[i][2], 4)
        #     tweets_data[i].proK3 = proK
        #     tweets_data[i].proM3 = proM
        #     tweets_data[i].proA3 = proA
            # print(y[i])

        y = clf2.predict_proba(v2.transform(X))
        for i in range(len(y)):
            proK = round(y[i][0], 4)
            proM = round(y[i][1], 4)
            tweets_data[i].proK = proK
            tweets_data[i].proM = proM

        sess.add_all(tweets_data)
        sess.commit()


def get_camp_hashtags():
    # print("Loaded camp hashtags.")
    sess = get_session()
    hts = sess.query(Camp_Hashtag)
    hts = [(ht.hashtag, ht.camp) for ht in hts]
    print(f"Loaded {len(hts)} camp hashtags.")
    sess.close()
    return hts


def get_retweets(sess, start, end):
    """
    获取某段时间的全部retweets
    """
    print(f"Get retweets from {start} to {end}")
    tweets = sess.query(Retweet).filter(
        Retweet.dt >= start,
        Retweet.dt < end).yield_per(5000)
    return tweets


def get_ori_users(sess, start, end, uid):
    """
    获取原始user_id
    """
    # print(f"Get retweets from {start} to {end}")
    tweets = sess.query(Retweet.ori_user_id).filter(
        Retweet.user_id == uid,
        Retweet.dt >= start,
        Retweet.dt < end).distinct()
    return tweets


def get_ori_users_v2(start, end, uids):
    """
    获取原始user_id

    __tablename__ = "retweets"
    tweet_id = Column(Integer, primary_key=True)
    dt = Column(DateTime)
    user_id = Column(Integer)
    ori_tweet_id = Column(Integer)
    ori_user_id = Column(Integer)

    """
    sess = get_session()
    friends_of_users = defaultdict(set)
    for t in tqdm(get_retweets(sess, start, end)):
        if t.user_id in uids:
            friends_of_users[t.user_id].add(t.ori_user_id)
    sess.close()
    return friends_of_users


def get_retweets_graph():
    """
    导入转发推特
    """
    import networkx as nx
    sess = get_session()
    count = 0
    # ALL
    g = nx.DiGraph()
    for r, _ in tqdm(sess.query(Retweet, Tweet).
                     filter(Retweet.tweet_id == Tweet.tweet_id).
                     filter(or_(Tweet.proM < 0.25, Tweet.proM > 0.75)).yield_per(5000)):
        count += 1
        g.add_edge(r.ori_user_id, r.user_id)
        if count >= 10000:
            break

    out_name = "data/network_ALL.gml"
    print("saving the graph ...", out_name)
    # nx.write_gpickle(g, out_name)
    nx.write_gml(g, out_name)
    return 0

    # Cristina
    g = nx.DiGraph()
    for r, t in tqdm(sess.query(Retweet, Tweet).
                     filter(Retweet.tweet_id == Tweet.tweet_id).
                     filter(Tweet.proM < 0.25).yield_per(5000)):

        g.add_edge(r.ori_user_id, r.user_id)

    out_name = "disk/data/network_K.gp"
    print("saving the graph ...", out_name)
    nx.write_gpickle(g, out_name)

    # Macri
    g = nx.DiGraph()
    for r, t in tqdm(sess.query(Retweet, Tweet).
                     filter(Retweet.tweet_id == Tweet.tweet_id).
                     filter(Tweet.proM > 0.75).yield_per(5000)):

        g.add_edge(r.ori_user_id, r.user_id)

    out_name = "disk/data/network_M.gp"
    print("saving the graph ...", out_name)
    nx.write_gpickle(g, out_name)


def get_all_tweets_75():
    sess = get_session()
    tweets = sess.query(Tweet.tweet_id).filter(
        Tweet.source.is_(None), Tweet.proM >= 0.75).yield_per(5000)
    with open("data/tweets-proM-0.75.txt", "w") as f:
        for t in tqdm(tweets):
            f.write(str(t[0]) + "\n")
    sess.close()


def get_all_tweets_25():
    sess = get_session()
    tweets = sess.query(Tweet.tweet_id).filter(
        Tweet.source.is_(None), Tweet.proM <= 0.25).yield_per(5000)
    with open("data/tweets-proM-0.25.txt", "w") as f:
        for t in tqdm(tweets):
            f.write(str(t[0]) + "\n")
    sess.close()


def get_all_tweets_with_hashtags(sess):
    tweets = sess.query(Tweet.tweet_id, Tweet.hashtags, Tweet.dt).filter(
        Tweet.hashtags.isnot(None)).yield_per(5000)
    return tweets


def get_tweets_with_hashtags(sess, start, end):
    tweets = sess.query(Tweet.tweet_id, Tweet.hashtags).filter(
        Tweet.hashtags.isnot(None),
        Tweet.source.is_(None),
        Tweet.dt >= start,
        Tweet.dt < end).yield_per(5000)
    return tweets


def get_tweets(sess, start, end, bots=False):
    """
    获取某段时间的全部tweets
    """
    print(f"Get tweets from {start} to {end}")
    if bots:
        tweets = sess.query(Tweet).filter(
            Tweet.source.isnot(None),
            Tweet.dt >= start,
            Tweet.dt < end).yield_per(10000)
    else:
        tweets = sess.query(Tweet).filter(
            Tweet.source.is_(None),
            Tweet.dt >= start,
            Tweet.dt < end).yield_per(10000)
    return tweets


def get_tweets_month(month):
    """
    obtain tweets of each month
    """
    sess  = get_session()
    start = pendulum.DateTime(2019, month, 1)
    end   = pendulum.DateTime(2019, month+1, 1)
    with open(f"disk/hernan/predicted_tweets_{start.format('YYYYMM')}.txt", "w") as f:
        f.write("tweet_id,user_id,datetime,probM\n")
        for t in get_tweets(sess, start, end):
            f.write(f"{t.tweet_id},{t.user_id},{t.dt},{t.proM}\n")
    sess.close()


def get_tweets3(sess, start, end):
    """
    获取某段时间的全部tweets
    """
    print(f"Get tweets3 from {start} to {end}")
    tweets = sess.query(Tweet3.tweet_id).filter(
        Tweet3.dt >= start,
        Tweet3.dt < end).yield_per(5000)
    return tweets


def get_tweets_day(sess, dt):
    """
    获取某天的全部tweets
    """
    print(f"Get tweets from in {dt}")
    tweets = sess.query(Tweet).filter(
        Tweet.source.is_(None),
        Tweet.dt >= dt,
        Tweet.dt < dt.add(days=1)).yield_per(5000)
    return tweets


def get_bot_tweets_day(sess, dt):
    """
    获取某天的全部tweets
    """
    print(f"Get bots tweets from in {dt}")
    tweets = sess.query(Tweet).filter(
        Tweet.source.isnot(None),
        Tweet.dt >= dt,
        Tweet.dt < dt.add(days=1)).yield_per(5000)
    return tweets


def get_all_users(sess, bots=False):
    """
    获取某天的全部tweets
    """
    print(f"Get all users from DB")
    if not bots:
        users = sess.query(User).all()
    else:
        users = sess.query(Bot_User).all()

    users = {u.user_id: u.first_camp for u in users}
    return users


def clients_stat():
    """
    统计clients
    """
    sess = get_session()
    tweets = sess.query(Source.source).yield_per(5000)

    from collections import Counter
    cnt = Counter()
    for t in tweets:
        cnt[t[0]] += 1

    json.dump(cnt.most_common(), open(
        "data/client_stat_2019-05-07.json", "w"), indent=2)
    sess.close()


def get_tweets_day_with_hashtags(sess, dt):
    """
    获取某天的全部tweets
    """
    tweets = sess.query(Tweet).filter(
        Tweet.source.is_(None),
        Tweet.hashtags.isnot(None),
        Tweet.dt >= dt,
        Tweet.dt < dt.add(days=1)).yield_per(5000)
    return tweets


def get_term_stat():
    sess = get_session()
    new_data = []
    for t in sess.query(Term).order_by(desc(Term.proK)):
        new_data.append([
            t.name,
            t.proK,
            t.proM,
            t.unclassified,
        ])
    sess.close()
    return new_data


def init_db():
    engine = create_engine("sqlite:///./data/election.db")
    Base.metadata.create_all(engine)


################## get section ##################
def get_session():
    engine = create_engine("sqlite:///./data/election.db")
    # engine = create_engine(
    #     "sqlite:////home/alex/kayzhou/Argentina_election/data/election_v2.db")
    DBSession = sessionmaker(bind=engine)
    session = DBSession()
    return session


def _update():
    sess = get_session()

    start = pendulum.datetime(2019, 8, 12)  # include this date
    # start = pendulum.datetime(2019, 3, 1)
    # end = pendulum.datetime(2019, 8, 27)  # not include
    end = pendulum.datetime(2019, 8, 19)  # not include

    # import last tweets
    # tweets_to_db_v2(sess, start, end)

    # hashtag
    # tweets_db_to_hashtags(sess, start, end)
    # tweets_db_to_hashtags75(sess, end)
    hashtag_cnt = count_of_hashtags(sess, start, end)
    json.dump(hashtag_cnt, open(f"data/{start.to_date_string}-{end.to_date_string}-hashtags-count.json", "w"), indent=2)
    # tweets_db_to_hashtags75_lastweek(sess, end)

    # stat
    # db_to_users(sess, start, end)
    # db_to_stat_predict(sess, start, end, clear=True)

    # bots
    # db_to_users(sess, start, end, bots=True)
    # db_to_stat_predict(sess, start, end, bots=True, clear=True)

    # 预测更新
    # predict_per_day(sess, start, end)
    # predict_per_week(sess, end)
    sess.close()


def predict_special_day(end, bots=True):
    # Since we lost data of PASO, from 5-7, we start the predict.
    start = pendulum.datetime(2019, 5, 7)
    sess = get_session()
    tweets = sess.query(Tweet).filter(
        Tweet.source.is_(None),
        Tweet.dt >= start,
        Tweet.dt < end).yield_per(5000)

    users = {}
    _dt = start
    for t in tqdm(tweets):
        uid = t.user_id
        if _dt not in users:
            users[_dt] = {}
        if uid not in users[_dt]:
            users[_dt][uid] = {
                "proM": 0,
                "proK": 0,
                "Unclassified": 0,
            }
        if t.proM > 0.75:
            users[_dt][uid]["proM"] += 1
        elif t.proM < 0.25:
            users[_dt][uid]["proK"] += 1
        else:
            users[_dt][uid]["Unclassified"] += 1

    users_v2 = []
    for dt, us in users.items():
        # print(dt)
        cnt = {
            "K": 0,
            "M": 0,
            "U": 0,
            "irrelevant": 0,
        }
        for u, v in us.items():
            if v["proM"] > v["proK"]:
                cnt["M"] += 1
            elif v["proM"] < v["proK"]:
                cnt["K"] += 1
            elif v["proM"] > 0 or v["proK"] > 0:
                cnt["U"] += 1
            else:
                cnt["irrelevant"] += 1

        print(dt, end, cnt)
        sess.add(Weekly_Predict(dt=end,
                                U_Cristina=cnt["K"],
                                U_Macri=cnt["M"],
                                U_unclassified=cnt["U"],
                                U_irrelevant=cnt["irrelevant"]))
        sess.commit()
        _sum = cnt["K"] + cnt["M"]
        print(cnt["K"] / _sum, cnt["M"] / _sum)
    sess.close()
    

if __name__ == "__main__":
    init_db()
    # _update()
    # 添加民调到db中
    # add_other_polls(clear=True)

    # get_all_tweets_25()
    # get_all_tweets_75()

    # tweets to database
    # sess = get_session()
    # start = pendulum.datetime(2019, 10, 11, tz="UTC") # include this date
    # end = pendulum.datetime(2019, 10, 27, tz="UTC") # not include this date
    # tweets_to_db_v2(sess, start, end, clear=True)
    # sess.close()
    # predict_Ndays(start, end, win=30, clear=True)

    # start = pendulum.datetime(2019, 10, 1, tz="UTC") # include this date
    # end = pendulum.datetime(2019, 10, 27, tz="UTC") # not include this date
    # save daily user snapshot with different probs
    # sess = get_session()
    # for dt in pendulum.Period(start, end):
        # save_today_user_snapshot_ignore(sess, dt, prob=0.68, ignore_id=ig_set)
        # save_today_user_snapshot(sess, dt, prob=0.6)
        # save_today_user_snapshot(sess, dt, prob=0.64)
        # save_today_user_snapshot(sess, dt, prob=0.66)
        # save_today_user_snapshot(sess, dt, prob=0.68)
        # save_today_user_snapshot(sess, dt, prob=0.72)
        # save_today_user_snapshot(sess, dt, prob=0.75)
    # sess.close()

    # sess = get_session()
    start = pendulum.datetime(2019, 3, 1, tz="UTC") # include this date
    end = pendulum.datetime(2019, 10, 27, tz="UTC") # not include this date
    # db_to_stat_predict_v2(sess, start, end)
    # sess.close()

    # window user file
    new_save_user_snapshot(start, end, w=7, prob=0.66)
    new_save_user_snapshot(start, end, w=14, prob=0.66)
    new_save_user_snapshot(start, end, w=21, prob=0.66)
    new_save_user_snapshot(start, end, w=28, prob=0.66)
    new_save_user_snapshot(start, end, w=56, prob=0.66)
    
    # cumulative prediction from March 1st
    """
    prob=0.66
    start = pendulum.datetime(2019, 3, 2, tz="UTC") # include this date
    end = pendulum.datetime(2019, 10, 27, tz="UTC") # not include this date
    copyfile(f"disk/users_v2/2019-03-01-{prob}.txt", f"disk/cul_from_March_1_v2/2019-03-02-{prob}.txt")
    predict_cumulative_file(start, end, prob=prob)
    predict_cumulative_to_csv(start, end, prob=prob, in_dir="from_March_1_v2")
    """
    
    # prob=0.72
    # start = pendulum.datetime(2019, 3, 2, tz="UTC") # include this date
    # end = pendulum.datetime(2019, 10, 11, tz="UTC") # not include this date
    # copyfile("disk/users_v2/2019-03-01-0.72.txt", "disk/cul_from_March_1_v2/2019-03-02-0.72.txt")
    # predict_cumulative_file(start, end, prob=prob)
    # predict_cumulative_to_csv(start, end, prob=prob, in_dir="from_March_1_v2")

    # prob=0.75
    # start = pendulum.datetime(2019, 3, 2, tz="UTC") # include this date
    # end = pendulum.datetime(2019, 10, 11, tz="UTC") # not include this date
    # copyfile("disk/users_v2/2019-03-01-0.75.txt", "disk/cul_from_March_1_v2/2019-03-02-0.75.txt")
    # predict_cumulative_file(start, end, prob=prob)
    # predict_cumulative_to_csv(start, end, prob=prob, in_dir="from_March_1_v2")

    # 不同的起始时间
    # prob=0.66
    # for i in range(4, 9):
    #     start = pendulum.datetime(2019, i, 2, tz="UTC") # include this date
    #     end = pendulum.datetime(2019, 10, 11, tz="UTC") # not include this date
    #     copyfile(f"disk/users_v2/2019-{i:02}-01-{prob}.txt", f"disk/cul_{i}/2019-{i:02}-02-{prob}.txt")
    #     start = start.add(days=1)
    #     predict_cumulative_file(start, end, prob=prob, out_dir=str(i))
    #     predict_cumulative_to_csv(start, end, prob=prob, in_dir=str(i))
        
    # predict_culmulative_swing_loyal(start, end, prob=0.68)
    # predict_culmulative_user_class(start, end, prob=0.68)
    # new_users_in_different_class(start, end, w=60, prob=0.68)
    
    # old_users_in_different_class(start, end, peri=2)
    # old_users_in_different_class(start, end, peri=3)
    # old_users_in_different_class(start, end, peri=5)

    # tweets_to_db(sess)
    # tweets_to_source(sess)
    # update_classify_results(sess)
    # tweets_to_new_clas(sess)
    # add_camp_hashtags(clear=True)
    # get_camp_hashtags()

    # 统计terms次数
    # terms_stat()
    # terms_month_stat()
    # clients_stat()
    # print(get_term_stat())
    # count_paso_hashtag()

    # 转发信息倒入数据库
    # start = pendulum.datetime(2019, 5, 1, tz="UTC") # include this date
    # end =   pendulum.datetime(2019, 10, 11, tz="UTC") # not include this date
    # print(f"{start} <= run < {end}")
    # sess = get_session()
    # tweets_to_retweets(sess, start, end, clear=True)
    # sess.close()

    # sess = get_session()
    # tweets_to_retweets_all(sess)
    # sess.close()

    # get_retweets_graph()

    # deal with lost PASO 处理PASO五月初事故
    # end = pendulum.datetime(2019, 5, 9)
    # predict_special_day(end)
    # end = pendulum.datetime(2019, 5, 10)
    # predict_special_day(end)
    # end = pendulum.datetime(2019, 5, 11)
    # predict_special_day(end)
    # end = pendulum.datetime(2019, 5, 12)
    # predict_special_day(end)
    # end = pendulum.datetime(2019, 5, 14)
    # predict_day(sess, end, bots=True)

    # count_paso_camp()
    # count_per_day()

    # tweets_db_to_hashtags(sess, clear=False)

    # get_hashtags75_v2(sess)
    # get_raw_tweets()

    # count_file_hashtag("disk/201905/201905-Alberto AND Fernandez.txt")
    # get_top_hashtags(sess)

    # save user snapshot per day （快照）
    # predict_user_snapshot(7)

    # For Hernan
    # for month in range(10, 11):
    #     get_tweets_month(month)
