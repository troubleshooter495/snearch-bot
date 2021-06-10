import os
from random import randint


class UserDB:
    def __init__(self):
        self.user_liked = dict()
        self.user_disliked = dict()
        self.user_used = dict()
        self.user_sent = dict()
        self.user_predict = dict()

    def add2liked(self, user, liked):
        if user not in self.user_liked:
            self.user_liked[user] = set()
        self.user_liked[user].add(liked)

    def add2disliked(self, user, disliked):
        if user not in self.user_disliked:
            self.user_disliked[user] = set()
        self.user_disliked[user].add(disliked)

    def add2used(self, user, used):
        if user not in self.user_used:
            self.user_used[user] = set()
        self.user_used[user].add(used)

    def add2sent(self, user, img):
        if user not in self.user_sent:
            self.user_sent[user] = set()
        self.user_sent[user].add(img)

    def add2predict(self, user, list_to_predict):
        if user not in self.user_predict:
            self.user_predict[user] = set()
        for i in list_to_predict:
            self.user_predict[user].add(i)

    def predict(self, user):
        if user not in self.user_predict or not self.user_predict[user]:
            return None
        return self.user_predict[user].pop()

    def deletefromliked(self, user, liked):
        try:
            self.user_liked[user].remove(liked)
        except KeyError:
            pass

    def deletefromdisliked(self, user, disliked):
        try:
            self.user_disliked[user].remove(disliked)
        except KeyError:
            pass

    def deletefromused(self, user, used):
        try:
            self.user_used[user].remove(used)
        except KeyError:
            pass

    def isok(self, user, img):
        f = 1
        if user in self.user_used:
            f = f and img not in self.user_used[user]
        if user in self.user_disliked:
            f = f and img not in self.user_disliked[user]
        return f

    def getlikes(self, user):
        return self.user_liked[user] if user in self.user_liked else []

    def reset(self, user):
        self.user_liked[user] = set()
        self.user_disliked[user] = set()
        self.user_used[user] = set()
        self.user_sent[user] = set()
        self.user_predict[user] = set()


class ServerDB:
    def __init__(self, folder='data/base',
                 rand_resize=lambda: randint(1, 10) < 4):
        self.paths = []
        self.cats = []
        self.msg_pic = dict()

        for cat in os.listdir(folder):
            folder1 = folder + "/" + cat
            if cat == '.DS_Store':
                continue
            for subcat in os.listdir(folder1):
                folder2 = folder1 + "/" + subcat
                self.cats.append([])
                if subcat == '.DS_Store':
                    continue
                for brand in os.listdir(folder2):
                    folder3 = folder2 + "/" + brand
                    if brand == '.DS_Store':
                        continue
                    for pic in os.listdir(folder3):
                        if rand_resize():
                            continue
                        self.paths.append(folder3 + "/" + pic)
                        self.cats[-1].append(folder3 + "/" + pic)

        self.cats = [t for t in self.cats if t != []]

        print("server db is loaded")

    def random(self):
        return self.paths[randint(0, len(self.paths) - 1)]

    def random_cat(self):
        cat = randint(0, len(self.cats) - 1)
        return self.cats[cat][randint(0, len(self.cats[cat]) - 1)]

    def add2msgpic(self, msgid, imgpath):
        self.msg_pic[msgid] = imgpath

    def imgfromchat(self, msgid):
        return self.msg_pic[msgid]

    def getimgpaths(self):
        return self.paths
