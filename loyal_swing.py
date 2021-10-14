# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    loyal_swing.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Kay Zhou <zhenkun91@outlook.com>           +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2019/10/22 16:52:31 by Kay Zhou          #+#    #+#              #
#    Updated: 2019/10/23 02:02:13 by Kay Zhou         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import ujson as json
import pendulum
from tqdm import tqdm
from collections import Counter
import pandas as pd


def save_union_users(p):
    """
    New for paper, the last k tweets, 2019-03-01 ~ 2019-10-10
    """
    all_users = {}
    for dt in tqdm(pendulum.Period(pendulum.Date(2019, 3, 1), pendulum.Date(2019, 10, 26))):
        print(dt)
        users = json.load(open(f"disk/users_v2/{dt.to_date_string()}-{p}.txt"))
        for u, v in users.items():
            v["dt"] = dt.to_date_string()
            if u not in all_users: # v 一天的tweets
                all_users[u] = [v]
            else:
                all_users[u].append(v)
    
    with open(f"data/users-20190301-20191026-{p}.json", "w") as f:
        for u, v in all_users.items():
            r = {}
            r["uid"] = u
            r["tweets"] = v
            f.write(json.dumps(r) + "\n")
    

def save_union_users_v2(p):
    """
    New for paper, the last k tweets, 2019-03-01 ~ 2019-10-10
    """
    import random

    k = 5 # 50
    out_file = open(f"data/users-20190301-20191026-opinion-5-{p}.json", "w")

    for line in tqdm(open(f"data/users-20190301-20191026-{p}.json")):
        d = json.loads(line.strip())
        uid = d["uid"]
        tweets = d["tweets"]

        last_k_tweets = [] # for short
        user_cum = {"FF": 0, "MP": 0} # for long

        user_rst = {"uid": uid, "opinion": []}

        for t in tweets: # 遍历一天的tweets

            if t["I"] > 0:
                user_rst["opinion"].append({
                    "dt": t["dt"], 
                    "long - short": "end"
                })
                break

            # for short
            today_tweets = []
            for i in range(t["K"]):
                today_tweets.append("K")
            for i in range(t["M"]):
                today_tweets.append("M")
            random.shuffle(today_tweets)

            if len(today_tweets) >= k:
                last_k_tweets = today_tweets
            else:
                last_k_tweets.extend(today_tweets)
                last_k_tweets = last_k_tweets[-k:]

            FF_MP_count = Counter(last_k_tweets)
            FF_count = FF_MP_count["K"]
            MP_count = FF_MP_count["M"]
            # print(FF_count, MP_count)
            if FF_count > MP_count:
                opinion_short = "FF"
            elif FF_count < MP_count:
                opinion_short = "MP"
            else:
                opinion_short = "Undecided"

            # for long
            user_cum["FF"] += t["K"]
            user_cum["MP"] += t["M"]

            if user_cum["FF"] > user_cum["MP"]:
                opinion_long = "loyal FF"
                if user_cum["MP"] == 0:
                    opinion_long = "Ultra loyal FF"
            elif user_cum["FF"] < user_cum["MP"]:
                opinion_long = "loyal MP"
                if user_cum["FF"] == 0:
                    opinion_long = "Ultra loyal MP"
            elif user_cum["FF"] == user_cum["MP"]:
                if user_cum["FF"] > 0:
                    opinion_long = "Undecided"
                else:
                    opinion_long = "Unclassified"

            if opinion_long.startswith("Ultra") or opinion_long.startswith("Unclassified"):
                opinion = opinion_long
            else:
                opinion = opinion_long + " - " + opinion_short

            # Opinion changes
            if user_rst["opinion"]:
                if user_rst["opinion"][-1]["long - short"] != opinion:
                    user_rst["opinion"].append({
                        "dt": t["dt"], 
                        "long - short": opinion
                    })
            else:
                user_rst["opinion"].append({
                    "dt": t["dt"], 
                    "long - short": opinion
                })

        # print(user_rst)
        out_file.write(json.dumps(user_rst) + "\n")    
            
            
def save_union_users_v3(p):
    """
    新需求，看最后k条tweets, 2019-03-01 ~ 2019-10-10
    """
    rsts = {}
    all_dates = [dt.to_date_string() for dt in pendulum.Period(pendulum.Date(2019, 3, 1),
                                               pendulum.Date(2019, 10, 27))]

    for dt in all_dates:
        rsts[dt] = {
            "Ultra loyal FF": 0,
            "Ultra loyal MP": 0,
            "loyal FF - FF": 0,
            "loyal FF - MP": 0,
            "loyal FF - Undecided": 0,
            "loyal MP - FF": 0,
            "loyal MP - MP": 0,
            "loyal MP - Undecided": 0,
            "Undecided - FF": 0,
            "Undecided - MP": 0,
            "Undecided - Undecided": 0,
            "Unclassified": 0,
        }
    # print(rsts)

    for line in tqdm(open(f"data/users-20190301-20191026-opinion-5-{p}.json")):
        d = json.loads(line.strip())
        # uid = d["uid"]
        opinion = d["opinion"]
        i = 0
        next_dt = opinion[i]["dt"]

        for j in range(len(all_dates) - 1):
            dt  = all_dates[j]
            tow = all_dates[j+1]

            if dt == next_dt: # opinion may change
                if i + 1 <= len(opinion) - 1:
                    next_dt = opinion[i+1]["dt"]
                op = opinion[i]["long - short"]
                if op == "end":
                    break
                i += 1

            if i > 0: # from the first opinion
                rsts[tow][op] += 1
                
    json.dump(rsts, open(f"data/users-20190302-20191027-opinion-5-ts-{p}.json", "w"))


def save_union_users_v4(p):
    """
    新需求，看最后k条tweets, 2019-03-01 ~ 2019-10-10
    """
    data = pd.read_json(f"data/users-20190302-20191027-opinion-5-ts-{p}.json")
    data = pd.DataFrame(data.values.T, index=data.columns, columns=data.index)
    data["FF"] = data["Ultra loyal FF"] + data["loyal FF - FF"] + data["loyal FF - MP"] + data["loyal FF - Undecided"]
    data["MP"] = data["Ultra loyal MP"] + data["loyal MP - FF"] + data["loyal MP - MP"] + data["loyal MP - Undecided"]
    data["Undecided"] = data["Undecided - FF"] + data["Undecided - MP"] + data["Undecided - Undecided"]
    data.to_excel(f"data/Instantaneous (k=5) and cumulative prediction_p={p}.xlsx")


if __name__ == "__main__":
    # for p in [0.68, 0.72, 0.75]:
    for p in [0.66]:
        # save_union_users(p)
        # save_union_users_v2(p)
        save_union_users_v3(p)
        save_union_users_v4(p)