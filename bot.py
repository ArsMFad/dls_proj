from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.types import InputFile

from aiogram.contrib.fsm_storage.memory import MemoryStorage


from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup

from aiogram.utils import executor
from aiogram.utils.executor import start_webhook


from config import TOKEN

import main
import asyncio
import threading

bot = Bot(token=TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())


class ImgStates(StatesGroup):
    waiting_for_content = State()
    waiting_for_style = State()
    waiting_for_quality = State()


@dp.message_handler(commands=['start'])
async def cmd_start(message: types.Message):
    await message.answer("Hola Amigo!")


@dp.message_handler(commands=['cancel'], state="*")
async def cmd_cancel(message: types.Message, state: FSMContext):
    print("cancel")
    await state.finish()
    await message.answer("Действие отменено")


@dp.message_handler(commands=['style_img'], state="*")
async def load_content(message: types.Message):
    await ImgStates.waiting_for_content.set()
    await message.answer("Give me content")


@dp.message_handler(state=ImgStates.waiting_for_content, content_types=['document'])
async def content_loaded(message: types.Message, state: FSMContext):
    file_id = message.document.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    await bot.download_file(file_path, "cont" + str(message.from_user.id) + ".jpg")
    await state.update_data(content_img="cont" + str(message.from_user.id) + ".jpg")
    await message.answer("Content loaded")
    await ImgStates.waiting_for_style.set()
    await message.answer("Give me style")


@dp.message_handler(state=ImgStates.waiting_for_style, content_types=['document'])
async def style_loaded(message: types.Message, state: FSMContext):
    file_id = message.document.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    await bot.download_file(file_path, "style" + str(message.from_user.id) + ".jpg")
    await state.update_data(style_img="style" + str(message.from_user.id) + ".jpg")
    await message.answer("Style loaded")
    await ImgStates.waiting_for_quality.set()
    await message.answer("Give me quality")


@dp.message_handler(state=ImgStates.waiting_for_quality)
async def quality_loaded(message: types.Message, state: FSMContext):
    await message.answer(message.text)
    await state.update_data(quality=message.text)
    await message.answer("Quality got")
    user_data = await state.get_data()
    await message.answer(user_data)
    await state.finish()
    await message.answer("Wait")
    t = threading.Thread(
        target=lambda message, content_img, style_img, quality:
        asyncio.run(process_nst(message, content_img, style_img, quality)),
        args=(message, user_data["content_img"], user_data["style_img"], user_data["quality"]))
    t.start()


async def process_nst(message, content_img, style_img, quality):
    main.return_result(content_img, style_img, message.from_user.id, quality)
    file = InputFile(str(message.from_user.id) + "result.jpg")
    bot1 = Bot(token=TOKEN)
    await bot1.send_message(message.chat.id, "Готово!")
    await bot1.send_photo(message.chat.id, photo=file)
    await bot1.close_bot()


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
