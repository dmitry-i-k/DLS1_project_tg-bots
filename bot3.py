""" Это основной модуль, работает с телеграммом, выводит меню, обрабатывает команды, сохраняет файлы,
    выводит результат.
    В модуле transfer_module весь функционал по обработке изображений реализован в классе
    подключается : from transfer_module import Transfer_class
    писал, используя примеры для начинающих
    получилось не оптимально и запутанно
    используются глобальные переменные
    в основном словари, ключи которых это chat.id:
    словарь с именами сохраненных изображений стиля(для каждого чата одно изображение)
    аналогично для контента
    такие же словари для настроек
    изображения хранятся, можно менять местами, можно обновлять
    можно менять настройки и заново запускать
    логика отображения клавиатур и обработки получилась неочевидная и запутанная
    (в циклы многое стоило поместить, другие структуры данных применить)
"""
import telebot
from telebot import types
import os
import urllib.request
from transfer_module import Transfer_class

#очистка папки с изображениями, и словарей с их адресами
def clean_temp(mypath):
    global styleimg_name, contentimg_name, last_instruction

    styleimg_name = {}
    contentimg_name = {}
    last_instruction = {}

    for root, dirs, files in os.walk(mypath):
        for file in files:
            os.remove(os.path.join(root, file))
#папка для сохранения загруженных изображений
result_storage_path = 'temp'
#почистим, если остались картинки
clean_temp(result_storage_path)
#сколько файлов загрузили (чтобы периодически удалять)
files_counter = 0
#инициализируем класс переноса изображений
transfer_object = Transfer_class()
#опции: веса стиля, разрешения
#512 локалтно работало, выдавало красивые картинки, но на хероку не хватает памяти
style_levels = (10_000, 10_000_000)
resolutions = (64, 128, 256)
#режимы по умолчанию
resolution_init = resolutions[0]
style_level_init = style_levels[0]
#словари для режимов, ключи - id чата. для кажлого чата свои файлы, свои настройки
style_level = {}
resolution = {}

setings = ()
#запускаем бот
TOKEN = '№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№№'
bot = telebot.TeleBot(TOKEN)


#обработчик команды старт
@bot.message_handler(commands=['start'])
def start_message(message):
    global last_instruction
    bot.send_message(message.chat.id, "Привет. Делаю перенос стиля с одного изображения на второе.")
    cid = message.chat.id
    #хранит последнюю нажатую кнопку, чтобы в следующий раз не отображать - не загромождать меню
    last_instruction[cid] = ''
    #задаем начальные опции
    resolution[cid] = resolution_init
    style_level[cid] = style_level_init
    #запускаем клавиатуру
    bot.send_message(cid, 'жми, чтобы начать', reply_markup=keyboard(cid, 'start'))

#создает клавиатуру в зависимости от глобальных переменных
def keyboard(cid, where_call):
    kb = types.InlineKeyboardMarkup()
    #вызов основной клавиатуры(ниже вложенные)
    if where_call == 'start':
        #если последними командами не были загрузка стиля или обмен картинок местами
        if last_instruction[cid] not in ['style', 'swap']:
            #добавляется кнопка "загружаем картинку со стилем"
            key_style = types.InlineKeyboardButton(text='Прислать картинку со стилем', callback_data='style')
            kb.add(key_style)
        # если последними командами не были загрузка контента или обмен картинок местами
        if last_instruction[cid] not in ['content', 'swap']:
            # добавляется кнопка
            key_content = types.InlineKeyboardButton(text='Прислать фото для обработки', callback_data='content')
            kb.add(key_content)
        #если обе картинки загружены - можно запустить алгоритм или настроить, можно местами стиль и контент поменять
        if cid in styleimg_name and cid in contentimg_name:
            if last_instruction[cid] != 'show':
                key_show = types.InlineKeyboardButton(text='Получить стильную картинку', callback_data='show')
                kb.add(key_show)
            if last_instruction[cid] != 'set':
                key_set = types.InlineKeyboardButton(text='Настройки', callback_data='set')
                kb.add(key_set)
            if last_instruction[cid] != 'swap':
                key_swap = types.InlineKeyboardButton(text='Поменять картинки местами', callback_data='swap')
                kb.add(key_swap)
        return kb
    #вложенная клавиатура - что хотим настроить?
    elif where_call == 'options':
        key_res = types.InlineKeyboardButton(text='разрешение картинки на выходе', callback_data='res')
        key_level = types.InlineKeyboardButton(text='насколько стильным должно быть изображение ',
                                               callback_data='level')
        kb.add(key_res)
        kb.add(key_level)
        return kb
    # вложенная нижнего уровня - выбор разрешения
    elif where_call == 'options_res':
        key_128 = types.InlineKeyboardButton(text=f'разрешение {resolutions[0]} - очень быстро', callback_data='0')
        key_256 = types.InlineKeyboardButton(text=f'разрешение {resolutions[1]} - не очень быстро', callback_data='1')
        key_512 = types.InlineKeyboardButton(text=f'разрешение {resolutions[2]} - медленно ', callback_data='2')
        kb.add(key_128)
        kb.add(key_256)
        kb.add(key_512)
        return kb
    # вложенная нижнего уровня - выбор веса стиля
    elif where_call == 'options_level':
        key_high = types.InlineKeyboardButton(text='очень стилизовано', callback_data='high')
        key_low = types.InlineKeyboardButton(text='не очень стилизовано', callback_data='low')
        kb.add(key_high)
        kb.add(key_low)
        return kb

# Обработчик нажатий на кнопки
@bot.callback_query_handler(func=lambda call: True)
def callback_worker(call):
    #последняя команда
    global last_instruction
    #опции
    global resolution
    global style_level
    #если меню вложенные, то не удаляем меню после нажатия кнопки
    close_menu_flag = True
    cid = call.message.chat.id
    #последняя команда
    last_instruction[cid] = call.data

    if call.data == 'style':
        bot.send_message(cid, "загрузи картинку со стилем");
        bot.register_next_step_handler(call.message, get_style)  #запускаем функцию для загрузки
    elif call.data == 'content':
        bot.send_message(cid, "загрузи картинку для обработки");
        bot.register_next_step_handler(call.message, get_content)#запускаем функцию для загрузки
    elif call.data == 'swap':
        bot.send_message(cid, "теперь картинка со стилем и фото поменялись местами");
        swap(call.message)#запускаем функцию для обмена
    elif call.data == 'show':
        bot.send_message(cid, "буду думать, жди");
        showimage(call.message);# запускаем обработку изображения и вывод результата
    #вложенные меню
    elif call.data == 'set':
        bot.edit_message_text(chat_id=cid, message_id=call.message.message_id,
                              text='выбери настройки', reply_markup=keyboard(cid, 'options'))
        close_menu_flag = False
    elif call.data == 'res':
        bot.edit_message_text(chat_id=cid, message_id=call.message.message_id,
                              text=f'выбери разрешение, сейчас {resolution[cid]}',
                              reply_markup=keyboard(cid, 'options_res'))
        close_menu_flag = False
    elif call.data == 'level':
        bot.edit_message_text(chat_id=cid, message_id=call.message.message_id,
                              text=f'выбери уровень, сейчас {"очень" if style_level[cid] == style_levels[1] else "не очень"}',
                              reply_markup=keyboard(cid, 'options_level'))
        close_menu_flag = False
    elif call.data in ['0', '1', '2']:
        resolution[cid] = resolutions[int(call.data)]
        print(resolution[cid])
        bot.send_message(cid, 'жми, чтобы продолжить',
                         reply_markup=keyboard(cid, 'start'))

    elif call.data in ['high', 'low']:
        style_level[cid] = style_levels[1] if call.data == "high" else style_levels[0]
        print(style_level[cid])
        bot.send_message(cid, 'жми, чтобы продолжить',
                         reply_markup=keyboard(cid, 'start'))

    #удалить последнее меню, если не вложенное
    if close_menu_flag:
        bot.edit_message_reply_markup(cid, call.message.message_id)

#загрузка стиля
def get_style(message):
    global styleimg_name
    cid = message.chat.id
    # проверить что пришло фото
    if message.content_type == 'photo':
        #вызываем функцию сохранения изображения, возвращает имя
        styleimg_name[cid] = save_image_from_message(message)
        bot.send_message(cid, 'Отлично, я сохранил твое изображение стиля!')
    else:
        #если не фото , то
        bot.send_message(cid, 'Нужно загрузить изображение со стилем!')
        #чтобы кнопка ввести стиль не пропадала
        last_instruction[cid] = ''
    #запускаем клавиатуру
    bot.send_message(cid, 'жми, чтобы продолжить', reply_markup=keyboard(cid, 'start'))

#загрузка стиля(аналогичная загрузкеконтента, нужно было сделать одну функцию)
def get_content(message):
    global contentimg_name
    cid = message.chat.id
    # проверить тип
    if message.content_type =='photo':
        contentimg_name[cid] = save_image_from_message(message)
        bot.send_message(cid, 'Отлично, я сохранил твое изображение для обработки!')
    else:
        bot.send_message(cid, 'Нужно загрузить изображение для обработки!')
        last_instruction[cid] = ''
    bot.send_message(cid, 'жми, чтобы продолжить', reply_markup=keyboard(cid, 'start'))

#сохраяняет изображение из сообщения в файл
def save_image_from_message(message):
    global files_counter
    image_id = get_image_id_from_message(message)
    file_path = bot.get_file(image_id).file_path
    image_url = "https://api.telegram.org/file/bot{0}/{1}".format(TOKEN, file_path)
    if not os.path.exists(result_storage_path):
        os.makedirs(result_storage_path)
    image_name = "{0}.jpg".format(image_id)
    urllib.request.urlretrieve(image_url, "{0}/{1}".format(result_storage_path, image_name))
    files_counter +=1
    return image_name

def get_image_id_from_message(message):
    return message.photo[len(message.photo) - 1].file_id

#обучение и вывод картинки
def showimage(message):

    cid = message.chat.id
    # запускает обучение, передав два пути к исходным файлам
    output_image = transfer_object('{0}/{1}'.format(result_storage_path, styleimg_name[cid]),
                                    '{0}/{1}'.format(result_storage_path, contentimg_name[cid]),
                                        style_level[cid], resolution[cid])
    # выводит полученный результат
    bot.send_photo(cid, output_image, 'Ура, получилось!')
    bot.send_message(cid, 'Попробуем еще разок? :)')
    # запускает клавиатуру
    bot.send_message(cid, 'жми, чтобы продолжить', reply_markup=keyboard(cid, 'start'))
    # запускает проверку/очистку накопившихся файлов
    clean_images()

#обмен картинок местами
def swap(message):
    cid = message.chat.id
    styleimg_name[cid], contentimg_name[cid] = contentimg_name[cid], styleimg_name[cid]
    bot.send_message(cid, 'жми, чтобы продолжить', reply_markup=keyboard(cid, 'start'))
#некрасивая функция, для периодического удаления изображений
def clean_images():
    global files_counter
    print(files_counter)
    #насобиралось >10 картинок от всех пользователей, проверим сумарный объем
    if files_counter > 10:
        total_size = 0
        for root, dirs, files in os.walk(result_storage_path):
            for file in files:
                total_size += os.path.getsize(os.path.join(root, file))
        #больше 30 мб - почистим
        if total_size > 3e+7:
            clean_temp(result_storage_path)
        #если мало весят, дальше собираем
        print(total_size)
        #счетчик обнуляем если чистили и если мелкие
        files_counter = 0

#опрашиваем телеграмм
bot.polling(none_stop=True)
