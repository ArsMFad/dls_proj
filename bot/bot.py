from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher

from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup

from aiogram.utils import executor

from config import TOKEN

import asyncio


async def set_commands(bot: Bot):
    commands = [
        types.BotCommand(command="/style_img",
                   description="Наложить стиль на картинку"),
        types.BotCommand(command="/cancel", description="Отмена"),
    ]

    await bot.set_my_commands(commands)


class ImgStates(StatesGroup):
    waiting_for_content = State()
    waiting_for_style = State()
    waiting_for_quality = State()
    waiting_for_intensivity = State()


async def cmd_start(message: types.Message):
    await message.answer("Hola Amigo!")


async def cmd_cancel(message: types.Message, state: FSMContext):
    await state.finish()
    await message.answer("Действие отменено")


async def load_content(message: types.Message):
    await message.answer("Give me content")
    await ImgStates.waiting_for_content.set()


async def content_loaded(message: types.Message, state: FSMContext):
    await message.photo[-1].download('cont' + str(message.from_user.id) + ".jpg")
    await state.update_data(content_img=message.photo[-1])
    await message.answer("Content loaded")
    await ImgStates.next()
    await message.answer("Give me style")


async def style_loaded(message: types.Message, state: FSMContext):
    await message.photo[-1].download('style' + str(message.from_user.id) + ".jpg")
    await state.update_data(style_img=message.photo[-1])
    await message.answer("Style loaded")
    await ImgStates.next()
    await message.answer("Give me quality")


async def quality_loaded(message: types.Message, state: FSMContext):
    await message.answer(message.text)
    await state.update_data(quality=message.text)
    await message.answer("Quality got")
    await ImgStates.next()
    await message.answer("Give me intensivity")


async def intensivity_loaded(message: types.Message, state: FSMContext):
    await message.answer(message.text)
    await state.update_data(intensivity=message.text)
    await message.answer("Intensivity got")
    user_data = await state.get_data()
    await message.answer(user_data)


def register_handlers_style_image(dp: Dispatcher):
    dp.register_message_handler(load_content, commands="style_img", state="*")
    dp.register_message_handler(content_loaded, state=ImgStates.waiting_for_content)
    dp.register_message_handler(style_loaded, state=ImgStates.waiting_for_style)
    dp.register_message_handler(quality_loaded, state=ImgStates.waiting_for_quality)
    dp.register_message_handler(intensivity_loaded, state=ImgStates.waiting_for_intensivity)


def register_handlers_common(dp: Dispatcher):
    dp.register_message_handler(cmd_start, commands="start")
    dp.register_message_handler(cmd_cancel, commands="cancel")


async def main():
    bot = Bot(token=TOKEN)
    dp = Dispatcher(bot)

    register_handlers_common(dp)
    register_handlers_style_image(dp)

    await set_commands(bot)

    await dp.start_polling()


if __name__ == '__main__':
    asyncio.run(main())
