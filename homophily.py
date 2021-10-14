from my_weapon import *
from SQLite_handler2 import *
import networkx as nx


def get_undecided_users(_users):
    _undecided_users = []
    _K_users = set()
    _M_users = set()
    for u, v in _users.items():
        u = int(u)
        if v["I"] > 0:
            continue
        if v["K"] == v["M"] and v["K"] > 0:
            _undecided_users.append(u)
        elif v["K"] > v["M"]:
            _K_users.add(u)
        elif v["M"] > v["K"]:
            _M_users.add(u)
            
    return _undecided_users, _K_users, _M_users


def get_unclassified_users(_users):
    _undecided_users = []
    _K_users = set()
    _M_users = set()
    for u, v in _users.items():
        u = int(u)
        if v["I"] > 0:
            continue
        if v["K"] == 0 and v["M"] == 0:
            _undecided_users.append(u)
        elif v["K"] > v["M"]:
            _K_users.add(u)
        elif v["M"] > v["K"]:
            _M_users.add(u)
            
    return _undecided_users, _K_users, _M_users


def get_retweets_graph_cumulative_noP():
    """
    get retweets cumulative graph from database
    """
    sess = get_session()
    start = pendulum.date(2019, 3, 1)
    end   = pendulum.date(2019, 10, 27)
    g = nx.DiGraph()
    for dt in tqdm(pendulum.Period(start, end)):
        next_day = dt.add(days=1)
        for r in tqdm(sess.query(Retweet).filter(Retweet.dt >= dt, Retweet.dt < next_day).yield_per(5000)):
            g.add_edge(r.ori_user_id, r.user_id)
        
        # if next_day.to_date_string() == "2019-08-11" or next_day.to_date_string() == "2019-10-11": 
        # if next_day.to_date_string() == "2019-10-27": 
        out_name = f"D:\\ARG2019\\network\\{next_day.to_date_string()}.gpickle"
        print("saving the graph ...", out_name)
        nx.write_gpickle(g, out_name)
    sess.close()


def get_retweets_graph_cumulative(p):
    """
    get retweets cumulative graph from database
    """
    sess = get_session()
    start = pendulum.date(2019, 3, 1)
    end   = pendulum.date(2019, 8, 10)
    g = nx.DiGraph()
    for dt in pendulum.Period(start, end):
        next_day = dt.add(days=1)
        for r, t in sess.query(Retweet, Tweet).filter(Retweet.dt >= dt, Retweet.dt < next_day). \
                                            filter(Retweet.tweet_id == Tweet.tweet_id). \
                                            filter(or_(Tweet.proM < 1 - p, Tweet.proM > p)). \
                                            yield_per(5000):
            g.add_edge(r.ori_user_id, r.user_id)
        out_name = f"new_disk/network/{next_day.to_date_string()}-{p}.gp"
        print("saving the graph ...", out_name)
        # if next_day.to_date_string() == "2019-08-11" or next_day.to_date_string() == "2019-10-11": 
        if next_day.to_date_string() == "2019-09-11": 
            nx.write_gpickle(g, out_name)
    sess.close()


def classify_undecided(p):
    out_file = open(f"data/homophily-{p}.json", "w")
    start = pendulum.date(2019, 3, 2)
    end   = pendulum.date(2019, 10, 27)
    for dt in pendulum.Period(start, end):
        # if dt.to_date_string() not in ["2019-10-27"]:
        #     continue
        # if dt.to_date_string() != "2019-09-11":
            # continue
        print(dt)
        FF = 0
        FF_users_homo = []
        MP = 0
        MP_users_homo = []
        UN = 0
        UN_users_homo = []
        
        users = json.load(open(f"disk/cul_from_March_1_v2/{dt.to_date_string()}-{p}.txt"))
        undecided_users, K_users, M_users = get_undecided_users(users)

        # graph = nx.read_gpickle(f"disk/network/{dt.to_date_string()}-{p}.gp")
        graph = nx.read_gpickle(f"new_disk/network/{dt.to_date_string()}.gp")
        graph = graph.to_undirected()
        
        for u in undecided_users:
            _k = 0
            _m = 0
            if u in graph:
                for neigh in graph.neighbors(u):
                    if neigh in K_users:
                        _k += 1
                    elif neigh in M_users:
                        _m += 1
            if _k > _m:
                FF += 1
                FF_users_homo.append(u)
            elif _k < _m:
                MP += 1
                MP_users_homo.append(u)
            else:
                UN += 1
                UN_users_homo.append(u)

        rst = {"dt": dt.to_date_string(), "FF": FF, "MP": MP, "UN": UN}
        print(rst)

        json.dump({"FF": FF_users_homo, "MP": MP_users_homo, "UN": UN_users_homo},
                   open(f"new_disk/homo_users/{dt.to_date_string()}-undecided-{p}.json", "w"))

        out_file.write(json.dumps(rst) + "\n")


def classify_unclassified(p):
    out_file = open(f"data/homophily-unclassified-{p}.json", "w")
    start = pendulum.date(2019, 3, 2)
    end   = pendulum.date(2019, 10, 11)
    for dt in pendulum.Period(start, end):
        if dt.to_date_string() not in ["2019-08-11", "2019-09-11", "2019-10-11", "2019-10-27"]:
            continue
        # if dt.to_date_string() != "2019-09-11":
            # continue
        print(dt)
        FF = 0
        FF_users_homo = []
        MP = 0
        MP_users_homo = []
        UN = 0
        UN_users_homo = []
        
        users = json.load(open(f"disk/cul_from_March_1_v2/{dt.to_date_string()}-{p}.txt"))
        undecided_users, K_users, M_users = get_unclassified_users(users)

        # graph = nx.read_gpickle(f"disk/network/{dt.to_date_string()}-{p}.gp")
        graph = nx.read_gpickle(f"new_disk/network/{dt.to_date_string()}.gp")
        graph = graph.to_undirected()
        
        for u in undecided_users:
            _k = 0
            _m = 0
            if u in graph:
                for neigh in graph.neighbors(u):
                    if neigh in K_users:
                        _k += 1
                    elif neigh in M_users:
                        _m += 1
            if _k > _m:
                FF += 1
                FF_users_homo.append(u)
            elif _k < _m:
                MP += 1
                MP_users_homo.append(u)
            else:
                UN += 1
                UN_users_homo.append(u)

        rst = {"dt": dt.to_date_string(), "FF": FF, "MP": MP, "UN": UN}
        print(rst)

        json.dump({"FF": FF_users_homo, "MP": MP_users_homo, "UN": UN_users_homo},
                   open(f"new_disk/homo_users/{dt.to_date_string()}-{p}.json", "w"))

        out_file.write(json.dumps(rst) + "\n")


if __name__ == "__main__":
    get_retweets_graph_cumulative_noP()
    # for p in [0.68, 0.72, 0.75]:
    for p in [0.66]:
        classify_undecided(p)
        # classify_unclassified(p)
