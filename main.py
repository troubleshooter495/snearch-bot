import telebot
import tok
import requests
import interface.classifier
from interface.classifier import Classifier
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import torch
import cv2
import interface.db
import interface.autoencoder
from interface.autoencoder import AE3
import interface.hnsw
from torchvision.transforms import transforms
import numpy as np

TOKEN = tok.token()
bot = telebot.TeleBot(TOKEN, parse_mode=None)
state = 0
serverdb = interface.db.ServerDB('data/base',
                                 lambda: np.random.randint(1, 10) < 4)
Classy = interface.classifier.load_classifier('data/classifier.pt')
userdb = interface.db.UserDB()
ae = interface.autoencoder.load_model('data/model.pt')
hnsw = interface.hnsw.Hnsw(serverdb.getimgpaths(), ae)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
classes = ('not shoe', 'shoe')
print('bot is loaded')


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message,
                 "Бот подбирает обувь, основываясь на ваших предпочтениях. "
                 "Каждой присланной паре можно поставить реакцию. "
                 "Доступны следующие команды:\n"
                 "/recommend – увидеть рекоммендацию обуви от бота\n"
                 "/send_photo – прислать фото понравившейся вам обуви\n")


@bot.message_handler(commands=['recommend'])
def recommend(message):
    user = message.from_user.id
    if not userdb.getlikes(user):
        print('RANDOM ONE')
        imgpath = serverdb.random_cat()
        while not userdb.isok(user, imgpath):
            imgpath = serverdb.random_cat()
    else:
        labels, dists = hnsw.knn(list(userdb.getlikes(user)), 5)
        f = 1
        for i in range(len(labels)):
            if userdb.isok(user, labels[i]):
                imgpath = labels[i]
                f = 0
                print(f'RECOMMENDED ONE i={i}')
                break

        if f:
            imgpath = serverdb.random_cat()
            print('COULD NOT RECOMMEND')


    img = open(imgpath, 'rb')
    m = bot.send_photo(message.chat.id, img, reply_markup=gen_markup())
    serverdb.add2msgpic(m.id, imgpath)
    userdb.add2used(user, imgpath)
    userdb.add2sent(user, imgpath)


@bot.message_handler(commands=['send_photo'])
def accept_photo(message):
    global state
    state = 1
    bot.reply_to(message, 'Теперь пришлите картинку')


@bot.callback_query_handler(func=lambda call: True)
def reaction(call):
    user = call.from_user.id
    img = serverdb.imgfromchat(call.message.id)

    if call.data == "like":
        userdb.add2liked(user, img)
        userdb.deletefromdisliked(user, img)
        bot.answer_callback_query(call.id, "liked")
    elif call.data == "dislike":
        userdb.add2disliked(user, img)
        userdb.deletefromliked(user, img)
        bot.answer_callback_query(call.id, "disliked")


@bot.message_handler(content_types=['photo'])
def accept_photo(message):
    global state
    if state == 0:
        return
    path = bot.get_file(message.photo[-1].file_id).file_path
    response = requests.get(f'https://api.telegram.org/file/bot{TOKEN}/{path}')
    imgpath = f"data/downloaded/{message.id}.png"

    with open(imgpath, "wb") as file:
        file.write(response.content)

    img = cv2.imread(imgpath)
    res = transform(cv2.resize(img, (32, 32)))
    pred = torch.argmax(Classy(res.view(1, 3, 32, 32)))

    if int(pred):
        userdb.add2liked(message.from_user.id, imgpath)
        bot.reply_to(message, 'Картинка успешно сохранена!')
    else:
        bot.reply_to(message, 'Не удалось распознать обувь на картинке')

    state = 0


def gen_markup():
    markup = InlineKeyboardMarkup()
    markup.row_width = 2
    markup.add(InlineKeyboardButton("👍", callback_data="like"),
               InlineKeyboardButton("👎", callback_data="dislike"))
    return markup


bot.polling()
