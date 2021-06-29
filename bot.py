from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.types import InputFile

from aiogram.contrib.fsm_storage.memory import MemoryStorage


from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup

from aiogram.utils import executor

from config import TOKEN

import os
import main
import asyncio
import threading

bot = Bot(token=TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())


class ImgStates(StatesGroup):
    waiting_for_content = State()
    waiting_for_style = State()
    waiting_for_model = State()
    waiting_for_quality = State()


@dp.message_handler(commands=['start'])
async def cmd_start(message: types.Message):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = ['StyleTransfer', 'Help']
    keyboard.add(*buttons)
    await message.answer("Hola Amigo!\nI can transfer style from one image (style) to another (content) (command StyleTransfer)\nIf you need help just type Help\nIf you want to cancel acion, just type 'Cancel'", reply_markup=keyboard)


@dp.message_handler(commands=['ancel'], state="*", commands_prefix='C')
async def cmd_cancel(message: types.Message, state: FSMContext):
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = ['StyleTransfer', 'Help']
    keyboard.add(*buttons)
    await state.finish()
    await message.answer("Action cancelled", reply_markup=keyboard)


@dp.message_handler(commands=['elp'], state="*", commands_prefix="H")
async def load_content(message: types.Message):
    await message.answer("I can transfer style from one image (style) to another (content).\nYou just need to type 'StyleTransfer'\nIF you want to cancel, just type 'Cancel'")


@dp.message_handler(commands=['tyleTransfer'], state="*", commands_prefix="S")
async def load_content(message: types.Message):
    await ImgStates.waiting_for_content.set()
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = ['Cancel']
    keyboard.add(*buttons)
    await message.answer("Send me content image AS FILE", reply_markup=keyboard)


@dp.message_handler(state=ImgStates.waiting_for_content, content_types=['document'])
async def content_loaded(message: types.Message, state: FSMContext):
    file_id = message.document.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    await bot.download_file(file_path, "cont" + str(message.from_user.id) + ".jpg")
    await state.update_data(content_img="cont" + str(message.from_user.id) + ".jpg")
    await message.answer("Content got")
    await ImgStates.waiting_for_style.set()
    await message.answer("Send me style image AS FILE")


@dp.message_handler(state=ImgStates.waiting_for_style, content_types=['document'])
async def style_loaded(message: types.Message, state: FSMContext):
    file_id = message.document.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    await bot.download_file(file_path, "style" + str(message.from_user.id) + ".jpg")
    await state.update_data(style_img="style" + str(message.from_user.id) + ".jpg")
    await message.answer("Style loaded")
    await ImgStates.waiting_for_model.set()
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = ["Daub", "Less daub"]
    keyboard.add(*buttons)
    await message.answer("Choose type of final result", reply_markup=keyboard)


@dp.message_handler(state=ImgStates.waiting_for_model)
async def style_loaded(message: types.Message, state: FSMContext):
    await message.answer(message.text)
    if message.text == ("Daub"):
        await state.update_data(model_choose="0")
        model_choose = "0"
    else:
        await state.update_data(model_choose="1")
        model_choose = "1"
    await ImgStates.waiting_for_quality.set()
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = ["Low (fast)", "Not low (not fast)", "High (slow)"]
    keyboard.add(*buttons)
    await message.answer("Choose quality of final result", reply_markup=keyboard)


@dp.message_handler(state=ImgStates.waiting_for_quality)
async def quality_loaded(message: types.Message, state: FSMContext):
    await message.answer(message.text)
    if message.text == ("Low (fast)"):
        await state.update_data(quality=1)
    elif message.text == ("Not low (not fast)"):
        await state.update_data(quality=2)
    else:
        await state.update_data(quality=3)

    await message.answer("Quality got", reply_markup=types.ReplyKeyboardRemove())
    user_data = await state.get_data()
    await state.finish()
    await message.answer("Wait for the result! (it usually takes about >5 minutes)")
    t = threading.Thread(
        target=lambda message, content_img, style_img, quality, model_choose:
        asyncio.run(process_nst(message, content_img, style_img, quality, model_choose)),
        args=(message, user_data["content_img"], user_data["style_img"], user_data["quality"], user_data["model_choose"]))
    t.start()


async def process_nst(message, content_img, style_img, quality, model_choose):
    main.return_result(content_img, style_img, message.from_user.id, quality, model_choose)
    file = InputFile(str(message.from_user.id) + "result.jpg")
    bot1 = Bot(token=TOKEN)
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    buttons = ["StyleTransfer", "Help"]
    keyboard.add(*buttons)
    await bot1.send_photo(message.chat.id, photo=file)
    await bot1.send_message(message.chat.id, "Готово!", reply_markup=keyboard)
    os.remove(content_img)
    os.remove(style_img)
    os.remove(str(message.from_user.id) + "result.jpg")
    await bot1.close_bot()


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
