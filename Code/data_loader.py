import random
import re
from collections import Counter

import numpy as np
import torch
from constants import (BATCH_SIZE, DATA_PATH, DESTINATION_LANG, EOS_TOKEN,
                       EOS_TOKEN_INDEX, SOS_TOKEN, SOS_TOKEN_INDEX,
                       SOURCE_LANG, TEST_RATIO, UKN_TOKEN, UKN_TOKEN_INDEX,
                       VALIDATION_RATIO, PAD_TOKEN, PAD_TOKEN_INDEX)
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from data_clean import clean_special_chars

# Data Ref :- 
# 1. https://tatoeba.org/en/downloads
# 2. https://drive.google.com/file/d/1xrD9bL78mbxpp-DdOw1EHhz1nzin_6dX/view


class LoadAndData:
    """
        Loads data
    """
    MAX_SENT_LEN = 0
    def __init__(self) -> None:
        
        self.source_path = f"{DATA_PATH}sample.{SOURCE_LANG}"
        self.destination_path = f"{DATA_PATH}sample.{DESTINATION_LANG}"

        self.destination = open(self.destination_path, encoding="utf-8").readlines()[:10000] * 333
        self.source = open(self.source_path, encoding="utf-8").readlines()[:10000] * 333

    @staticmethod
    def clean_sent(text):
        text = text.lower()
        text = text.replace('"', ' ').replace('”', '').replace("  ా ", " ").replace(" ఞ ", " ").replace("  ౖ ", " ").replace(" ၔ ", " ").replace("  ి ", " ").replace("  ొ ", " ").replace("  ే ", " ").replace("  ో ", " ").replace("  ై ", " ").replace("ϐ", "").replace("0", "").replace("🌅", "").replace("ଅ", "").replace("📱", "").replace("Δ", "").replace("λ", "").replace("△", "").replace("\u2061", "").replace("Ο", "").replace("\uf601", "").replace("–", "").replace("も", "").replace("ɪ", "").replace("≡", "").replace("ə", "").replace("☘", "").replace("Å", "").replace("Ϙ", "").replace("🔔", "").replace("¾", "").replace("✭", "").replace("ʃ", "").replace("Θ", "").replace("\\xa0", "").replace("👌", "").replace("⊖", "").replace("Ø", "").replace("”", "").replace("Ê", "").replace("µ", "").replace("τ", "").replace("✱", "").replace("つ", "").replace("Æ", "").replace("❁", "").replace("▾", "").replace("◀", "").replace("జ", "").replace("—", "").replace("ల", "").replace("ఽ", "").replace("é", "").replace("˚", "").replace("Φ", "").replace("Ϝ", "").replace("ó", "").replace("\uf0b7", "").replace("·", "").replace("∘", "").replace("ℎ", "").replace("⁂", "").replace("Σ", "").replace("‡", "").replace("\n", " ").replace("\u200c", "").replace("రూపాయలు", " రూపాయలు").replace("  ", " ").replace(" ట ", " ").replace(" ప ", " ").replace(" డ ", " ").replace(" మ ", " ").replace(" ా ", " ").replace(" ర ", " ").replace(" ఊ ", " ").replace(" జ ", " ").replace(" ో ", " ").replace(" హ ", " ").replace(" య ", " ").replace(" స ", " ").replace(" ఉ ", " ").replace(" చ ", " ").replace(" ం ", " ").replace(" ణ ", " ").replace(" ఒ ", " ").replace(" ఈ ", " ").replace(" ఏ ", " ").replace(" ఆ ", " ").replace(" వ ", " ").replace(" ఓ ", " ").replace(" ఎ ", " ").replace(" ల ", " ").replace(" న ", " ").replace(" అ ", " ").replace(" ఐ ", " ").replace(" ూ ", " ").replace(" త ", " ").replace(" ఇ ", " ").replace(" క ", " ").replace(" శ ", " ").replace(" ద ", " ").replace(" బ ", " ").replace(" ఘ ", " ").replace(" ఁ ", " ").replace(" ృ ", " ").replace(" ధ ", " ").replace(" ే ", " ").replace(" భ ", " ").replace(" ళ ", " ").replace(" థ ", " ").replace(" ౄ ", " ").replace(" ఔ ", " ").replace(" ః ", " ").replace(" ఫ ", " ").replace(" ె ", " ").replace(" ై ", " ").replace(" ి ", " ").replace(" ఖ ", " ").replace(" ్ ", " ").replace(" ీ ", " ").replace(" ష ", " ").replace(" ు ", " ").replace(" ొ ", " ").replace(" क ", " ").replace(" ౯ ", " ").replace(" ఠ ", " ").replace(" ఱ ", " ").replace(" ఝ ", " ").replace(" ౦ ", " ").replace(" ఋ ", " ").replace(" ౌ ", " ").replace(" ఌ ", " ").replace(" గ ", " ").replace(" ఛ ", " ").replace(" ఢ ", " ").replace(" ౨ ", " ").replace("✦", "").replace("’", "").replace("»", "").replace("►", "").replace("“", "").replace("‘", "").replace("₹", " రూపాయలు ").replace("▪", "").replace("⇒", "").replace("✺", "").replace("⍟", "").replace("♦", "").replace("¶", "").replace("°", "").replace("👇", "").replace("…", "").replace("阎", "").replace("ü", "").replace("½", "").replace("◆", "").replace("ƛ", "").replace("౩", "").replace("□", "").replace("±", "").replace("✔", "").replace("№", "").replace("★", "").replace("©", "").replace("❑", "").replace("≈", "").replace("•", "").replace("∠", "").replace("∫", "").replace("−", "").replace("౭", "").replace("§", "").replace("ê", "").replace("⇔", "").replace("×", "").replace("÷", "").replace("μ", "").replace("✪", "").replace("€", "").replace("→", "").replace("♪", "").replace("❏", "").replace("ε", "").replace("❂", "").replace("✘", "").replace("➦", "").replace("∇", "").replace("⇨", "").replace("≤", "").replace("α", "").replace("⛳", "").replace("✍", "").replace("∑", "").replace("॥", "").replace("£", "").replace("❤", "").replace("․", "").replace("β", "").replace("γ", "").replace("🏏", "").replace("∼", "").replace("ω", "").replace("¤", "").replace("‑", "").replace("😉", "").replace("∧", "").replace("➥", "").replace("⦿", "").replace("θ", "").replace("➤", "").replace("ø", "").replace("౬", "").replace("🙏", "").replace("।", "").replace("❝", "").replace("■", "").replace("æ", "").replace("י", "").replace("ρ", "").replace("σ", "").replace("و", "").replace("😶", "").replace("™", "").replace("✓", "").replace("✧", "").replace("à", "").replace("🦊", "").replace("≥", "").replace("π", "").replace("↗", "").replace("ā", "").replace("ϕ", "").replace("←", "").replace("❖", "").replace("●", "").replace("✸", "").replace("å", "").replace("౧", "").replace("¼", "").replace("❉", "").replace("✖", "").replace("²", "").replace("›", "").replace("¬", "").replace("薩", "").replace("旦", "").replace("ο", "").replace("➢", "").replace("∆", "").replace("○", "").replace("φ", "").replace("«", "").replace("♥", "").replace("✶", "").replace("👉", "").replace("✹", "").replace("✫", "").replace("ℓ", "").replace("⦾", "").replace("ン", "").replace("🔥", "").replace("☀", "").replace("∗", "").replace("δ", "").replace("✯", "").replace("ה", "").replace("☺", "").replace("⁄", "").replace("↓", "").replace("≠", "").replace("¹", "").replace("✩", "").replace("⦁", "").replace("⤏", "").replace("‰", "").replace("⚝", "").replace("🌾", "").replace("‣", "").replace("♣", "").replace("¸", "").replace("ː", "").replace("🔎", "").replace("¿", "").replace("😬", "").replace("◘", "").replace("✼", "").replace("∙", "").replace("⇩", "").replace("³", "").replace("🙂", "").replace("☆", "").replace("¡", "").replace("⚹", "").replace("❋", "").replace("®", "").replace("∞", "").replace("😟", "").replace("✴", "").replace("∪", "").replace("∩", "").replace("❃", "").replace("▶", "").replace("ä", "").replace("ö", "").replace("ß", "").replace("⊗", "").replace("ఙ", "").replace("¥", "").replace("▸", "").replace("в", "").replace("ц", "").replace("౫", "").replace("↑", "").replace("′", "").replace("\u200b", "").replace("\ufeff", "").replace("\uf0d8", "").replace("\uf642", "").replace("\u200d", "").replace("\uf0fc", "").replace("\uf02a", "").replace("\u2060", "").replace("\u200e", "").replace("\uf609", "").replace("\uf449", "").replace("\uf33a", "").replace("\uf60a", "").replace("1", "one ").replace("2", "two ").replace("3", " three ").replace("4", " four ").replace("5", " five ").replace("6", " six ").replace("7", " seven ").replace("8", " eight ").replace("9", " nine ").replace("`", " ").replace(" a ", " ").replace(" i ", " ").replace(" k ", " ").replace(" x ", " ").replace(" s ", " ").replace(" v ", " ").replace(" o ", " ").replace(" y ", " ").replace(" r ", " ").replace(" u ", " ").replace(" c ", " ").replace(" n ", " ").replace(" m ", " ").replace(" e ", " ").replace(" 0 ", " ").replace(" b ", " ").replace(" g ", " ").replace(" t ", " ").replace(" d ", " ").replace(" j ", " ").replace(" h ", " ").replace(" p ", " ").replace(" l ", " ").replace(" w ", " ").replace(" f ", " ").replace(" q ", " ").replace(" z ", " ").replace("'", '')
        # file.write(text + "\n")  

        text = text.replace('"', ' ')
        text = text.replace('”', '')

        text = text.replace("  ా ", " ")
        text = text.replace(" ఞ ", " ")
        text = text.replace("  ౖ ", " ")
        text = text.replace(" ၔ ", " ")
        text = text.replace("  ి ", " ")
        text = text.replace("  ొ ", " ")
        text = text.replace("  ే ", " ")
        text = text.replace("  ో ", " ")
        text = text.replace("  ై ", " ")
        text = text.replace("ϐ", "")
        text = text.replace("0", "")
        text = text.replace("🌅", "")
        text = text.replace("ଅ", "")
        text = text.replace("📱", "")
        text = text.replace("Δ", "")
        text = text.replace("λ", "")
        text = text.replace("△", "")
        text = text.replace("\u2061", "")
        text = text.replace("Ο", "")
        text = text.replace("\uf601", "")
        text = text.replace("–", "")
        text = text.replace("も", "")
        text = text.replace("ɪ", "")
        text = text.replace("≡", "")
        text = text.replace("ə", "")
        text = text.replace("☘", "")
        text = text.replace("Å", "")
        text = text.replace("Ϙ", "")
        text = text.replace("🔔", "")
        text = text.replace("¾", "")
        text = text.replace("✭", "")
        text = text.replace("ʃ", "")
        text = text.replace("Θ", "")
        text = text.replace("\\xa0", "")
        text = text.replace("👌", "")
        text = text.replace("⊖", "")
        text = text.replace("Ø", "")
        text = text.replace("”", "")
        text = text.replace("Ê", "")
        text = text.replace("µ", "")
        text = text.replace("τ", "")
        text = text.replace("✱", "")
        text = text.replace("つ", "")
        text = text.replace("Æ", "")
        text = text.replace("❁", "")
        text = text.replace("▾", "")
        text = text.replace("◀", "")
        text = text.replace("జ", "")
        text = text.replace("—", "")
        text = text.replace("ల", "")
        text = text.replace("ఽ", "")
        text = text.replace("é", "")
        text = text.replace("˚", "")
        text = text.replace("Φ", "")
        text = text.replace("Ϝ", "")
        text = text.replace("ó", "")
        text = text.replace("\uf0b7", "")
        text = text.replace("·", "")
        text = text.replace("∘", "")
        text = text.replace("ℎ", "")
        text = text.replace("⁂", "")
        text = text.replace("Σ", "")
        text = text.replace("‡", "")
        text = text.replace("\n", " ")
        text = text.replace("\u200c", "")
        text = text.replace("  ", " ")

        text = text.replace("✦", "")
        text = text.replace("’", "")
        text = text.replace("»", "")
        text = text.replace("►", "")
        text = text.replace("“", "")
        text = text.replace("‘", "")
        text = text.replace("▪", "")
        text = text.replace("⇒", "")
        text = text.replace("✺", "")
        text = text.replace("⍟", "")
        text = text.replace("♦", "")
        text = text.replace("¶", "")
        text = text.replace("°", "")
        text = text.replace("👇", "")
        text = text.replace("…", "")
        text = text.replace("阎", "")
        text = text.replace("ü", "")
        text = text.replace("½", "")
        text = text.replace("◆", "")
        text = text.replace("ƛ", "")
        text = text.replace("౩", "")
        text = text.replace("□", "")
        text = text.replace("±", "")
        text = text.replace("✔", "")
        text = text.replace("№", "")
        text = text.replace("★", "")
        text = text.replace("©", "")
        text = text.replace("❑", "")
        text = text.replace("≈", "")
        text = text.replace("•", "")
        text = text.replace("∠", "")
        text = text.replace("∫", "")
        text = text.replace("−", "")
        text = text.replace("౭", "")
        text = text.replace("§", "")
        text = text.replace("ê", "")
        text = text.replace("⇔", "")
        text = text.replace("×", "")
        text = text.replace("÷", "")
        text = text.replace("μ", "")
        text = text.replace("✪", "")
        text = text.replace("€", "")
        text = text.replace("→", "")
        text = text.replace("♪", "")
        text = text.replace("❏", "")
        text = text.replace("ε", "")
        text = text.replace("❂", "")
        text = text.replace("✘", "")
        text = text.replace("➦", "")
        text = text.replace("∇", "")
        text = text.replace("⇨", "")
        text = text.replace("≤", "")
        text = text.replace("α", "")
        text = text.replace("⛳", "")
        text = text.replace("✍", "")
        text = text.replace("∑", "")
        text = text.replace("॥", "")
        text = text.replace("£", "")
        text = text.replace("❤", "")
        text = text.replace("․", "")
        text = text.replace("β", "")
        text = text.replace("γ", "")
        text = text.replace("🏏", "")
        text = text.replace("∼", "")
        text = text.replace("ω", "")
        text = text.replace("¤", "")
        text = text.replace("‑", "")
        text = text.replace("😉", "")
        text = text.replace("∧", "")
        text = text.replace("➥", "")
        text = text.replace("⦿", "")
        text = text.replace("θ", "")
        text = text.replace("➤", "")
        text = text.replace("ø", "")
        text = text.replace("౬", "")
        text = text.replace("🙏", "")
        text = text.replace("।", "")
        text = text.replace("❝", "")
        text = text.replace("■", "")
        text = text.replace("æ", "")
        text = text.replace("י", "")
        text = text.replace("ρ", "")
        text = text.replace("σ", "")
        text = text.replace("و", "")
        text = text.replace("😶", "")
        text = text.replace("™", "")
        text = text.replace("✓", "")
        text = text.replace("✧", "")
        text = text.replace("à", "")
        text = text.replace("🦊", "")
        text = text.replace("≥", "")
        text = text.replace("π", "")
        text = text.replace("↗", "")
        text = text.replace("ā", "")
        text = text.replace("ϕ", "")
        text = text.replace("←", "")
        text = text.replace("❖", "")
        text = text.replace("●", "")
        text = text.replace("✸", "")
        text = text.replace("å", "")
        text = text.replace("౧", "")
        text = text.replace("¼", "")
        text = text.replace("❉", "")
        text = text.replace("✖", "")
        text = text.replace("²", "")
        text = text.replace("›", "")
        text = text.replace("¬", "")
        text = text.replace("薩", "")
        text = text.replace("旦", "")
        text = text.replace("ο", "")
        text = text.replace("➢", "")
        text = text.replace("∆", "")
        text = text.replace("○", "")
        text = text.replace("φ", "")
        text = text.replace("«", "")
        text = text.replace("♥", "")
        text = text.replace("✶", "")
        text = text.replace("👉", "")
        text = text.replace("✹", "")
        text = text.replace("✫", "")
        text = text.replace("ℓ", "")
        text = text.replace("⦾", "")
        text = text.replace("ン", "")
        text = text.replace("🔥", "")
        text = text.replace("☀", "")
        text = text.replace("∗", "")
        text = text.replace("δ", "")
        text = text.replace("✯", "")
        text = text.replace("ה", "")
        text = text.replace("☺", "")
        text = text.replace("⁄", "")
        text = text.replace("↓", "")
        text = text.replace("≠", "")
        text = text.replace("¹", "")
        text = text.replace("✩", "")
        text = text.replace("⦁", "")
        text = text.replace("⤏", "")
        text = text.replace("‰", "")
        text = text.replace("⚝", "")
        text = text.replace("🌾", "")
        text = text.replace("‣", "")
        text = text.replace("♣", "")
        text = text.replace("¸", "")
        text = text.replace("ː", "")
        text = text.replace("🔎", "")
        text = text.replace("¿", "")
        text = text.replace("😬", "")
        text = text.replace("◘", "")
        text = text.replace("✼", "")
        text = text.replace("∙", "")
        text = text.replace("⇩", "")
        text = text.replace("³", "")
        text = text.replace("🙂", "")
        text = text.replace("☆", "")
        text = text.replace("¡", "")
        text = text.replace("⚹", "")
        text = text.replace("❋", "")
        text = text.replace("®", "")
        text = text.replace("∞", "")
        text = text.replace("😟", "")
        text = text.replace("✴", "")
        text = text.replace("∪", "")
        text = text.replace("∩", "")
        text = text.replace("❃", "")
        text = text.replace("▶", "")
        text = text.replace("ä", "")
        text = text.replace("ö", "")
        text = text.replace("ß", "")
        text = text.replace("⊗", "")
        text = text.replace("ఙ", "")
        text = text.replace("¥", "")
        text = text.replace("▸", "")
        text = text.replace("в", "")
        text = text.replace("ц", "")
        text = text.replace("౫", "")
        text = text.replace("↑", "")
        text = text.replace("′", "")
        text = text.replace("\u200b", "")
        text = text.replace("\ufeff", "")
        text = text.replace("\uf0d8", "")
        text = text.replace("\uf642", "")
        text = text.replace("\u200d", "")
        text = text.replace("\uf0fc", "")
        text = text.replace("\uf02a", "")
        text = text.replace("\u2060", "")
        text = text.replace("\u200e", "")
        text = text.replace("\uf609", "")
        text = text.replace("\uf449", "")
        text = text.replace("\uf33a", "")
        text = text.replace("\uf60a", "")
        text = text.replace("1", " one ")
        text = text.replace("2", " two ")
        text = text.replace("3", " three ")
        text = text.replace("4", " four ")
        text = text.replace("5", " five ")
        text = text.replace("6", " six ")
        text = text.replace("7", " seven ")
        text = text.replace("8", " eight ")
        text = text.replace("9", " nine ")
        text = text.replace("౦", " సున్న ")
        text = text.replace("౧", " ఒకటి ")
        text = text.replace("౨", " రెండు ")
        text = text.replace("౩", " మూడు ")
        text = text.replace("౪", " నాలుగు ")
        text = text.replace("౫", " అయిదు ")
        text = text.replace("౬", " ఆరు ")
        text = text.replace("౭", " ఏడు ")
        text = text.replace("౮", " ఎనిమిది ")
        text = text.replace("౯", " తొమ్మిది ")

        text = text.replace(" a ", " ")
        text = text.replace(" i ", " ")
        text = text.replace(" k ", " ")
        text = text.replace(" x ", " ")
        text = text.replace(" s ", " ")
        text = text.replace(" v ", " ")
        text = text.replace(" o ", " ")
        text = text.replace(" y ", " ")
        text = text.replace(" r ", " ")
        text = text.replace(" u ", " ")
        text = text.replace(" c ", " ")
        text = text.replace(" n ", " ")
        text = text.replace(" m ", " ")
        text = text.replace(" e ", " ")
        text = text.replace(" 0 ", " ")
        text = text.replace(" b ", " ")
        text = text.replace(" g ", " ")
        text = text.replace(" t ", " ")
        text = text.replace(" d ", " ")
        text = text.replace(" j ", " ")
        text = text.replace(" h ", " ")
        text = text.replace(" p ", " ")
        text = text.replace(" l ", " ")
        text = text.replace(" w ", " ")
        text = text.replace(" f ", " ")
        text = text.replace(" q ", " ")
        text = text.replace(" z ", " ")
        text = text.replace("౮", "")
        
        text = text.replace("ו", "")
        text = text.replace("ם", "")
        text = text.replace("º", "")
        text = text.replace("پ", "")
        text = text.replace("😐", "")
        text = text.replace("ਏ", "")
        text = text.replace("ι", "")
        text = text.replace("ç", "")
        text = text.replace("ί", "")
        text = text.replace("क", "")
        text = text.replace("♠", "")
        text = text.replace("瞿", "")
        text = text.replace("桜", "")
        text = text.replace("大", "")
        text = text.replace("✷", "")
        text = text.replace("ή", "")
        text = text.replace("ʔ", "")
        text = text.replace("¨", "")
        text = text.replace("υ", "",)
        text = text.replace("ド", "",)
        text = text.replace("₣", "",)
        text = text.replace("塞", "",)
        text = text.replace("∂", "",)
        text = text.replace("\xad", "",)
        text = text.replace("ῥ", "",)
        text = text.replace("ĕ", "",)
        text = text.replace("ὰ", "",)
        text = text.replace("ῖ", "",)
        text = text.replace("ד", "",)
        text = text.replace("ش", "",)
        text = text.replace("ς", "",)
        text = text.replace("損", "",)
        text = text.replace("荆", "",)
        text = text.replace("ϝ", "",)
        text = text.replace("ῦ", "",)
        text = text.replace("ῆ", "",)
        text = text.replace("è", "",)
        text = text.replace("₂", "",)
        text = text.replace("ἰ", "",)
        text = text.replace("√", "",)
        text = text.replace("🏼", "",)
        text = text.replace("⅓", "",)
        text = text.replace("η", "",)
        text = text.replace("ϙ", "",)
        text = text.replace("\xa0", "",)
        text = text.replace("˙", "",)
        text = text.replace("ν", "",)
        text = text.replace("😞", "",)
        text = text.replace("‐", "",)
        text = text.replace("戀", "",)
        text = text.replace("℅", "",)
        text = text.replace("د", "",)
        text = text.replace("情", "",)
        text = text.replace("έ", "",)
        text = text.replace("那", "",)
        text = text.replace("χ", "",)
        text = text.replace("ά", "",)
        text = text.replace("中", "",)
        text = text.replace("愛", "",)
        text = text.replace("印", "",)
        text = text.replace("س", "",)
        text = text.replace("ر", "",)
        text = text.replace("イ", "",)
        text = text.replace("¢", "",)

        text = text.replace("\n", " ")
        text = text.replace("'", '')
        text = text.replace("`", " ")

        text = clean_special_chars(text)
        text = re.sub(r"\s+", " ", text)
        return text

    def run(self):
        self.source_clean = [self.clean_sent(i) for i in self.source]
        self.destination_clean = [self.clean_sent(i) for i in self.destination]

        with open(self.source_path.replace("sample", "clean_sample"), "w") as f:
            f.write("\n".join(self.source_clean))
        
        with open(self.destination_path.replace("sample", "clean_sample"), "w") as f:
            f.write("\n".join(self.destination_clean))
        return self.source_clean, self.destination_clean

    def max_sent_len(self, ):
        
        
        max_source_len = int(np.median(list(len(i.split(" ")) for i in self.source_clean)))
        max_dest_len = int(np.median(list(len(i.split(" ")) for i in self.destination_clean)))
        max_sent_len = int(np.mean([max_source_len, max_dest_len]))
        LoadAndData.MAX_SENT_LEN = max_sent_len
        return max_sent_len

    def train_data(self):

        self.data = list(zip(self.source_clean, self.destination_clean))
        random.shuffle(self.data)

        total_len = len(self.data)
        test_range = int(total_len * TEST_RATIO)
        validation_range = test_range + int(total_len * VALIDATION_RATIO)

        self.total_inds = [i for i in range(total_len)]
        random.shuffle(self.total_inds)
        self.test_inds = self.total_inds[:test_range]
        self.validate_inds = self.total_inds[test_range:validation_range]
        self.train_inds = self.total_inds[validation_range:]

        logger.info(f"total data points : {len(self.total_inds)}")
        logger.info(f"test range : {test_range}")
        logger.info(f"validation range : {validation_range}")

        logger.info(f"train data size : {len(self.train_inds)}")
        logger.info(f"test data size : {len(self.test_inds)}")
        logger.info(f"validation data size : {len(self.validate_inds)}")

        train_source = self.source_clean[validation_range:]
        train_destination = self.destination_clean[validation_range:]

        return train_source, train_destination

    def build_vocab(self, ):
        self.source_vocab = Counter()
        self.destination_vocab = Counter()

        train_source, train_destination = self.train_data()
        for s in train_source:
            self.source_vocab.update(s.strip().split(" "))
            
        for d in train_destination:
            self.destination_vocab.update(d.strip().split(" "))


        
        self.source_vocab_list = [SOS_TOKEN, EOS_TOKEN, UKN_TOKEN, PAD_TOKEN]
        self.destination_vocab_list = [SOS_TOKEN, EOS_TOKEN, UKN_TOKEN, PAD_TOKEN]
        
        # self.source_vocab_list[SOS_TOKEN_INDEX] = [SOS_TOKEN]
        # self.destination_vocab_list[SOS_TOKEN_INDEX] = [SOS_TOKEN]
        
        # self.source_vocab_list[EOS_TOKEN_INDEX] = [EOS_TOKEN]
        # self.destination_vocab_list[EOS_TOKEN_INDEX] = [EOS_TOKEN]
        
        # self.source_vocab_list[UKN_TOKEN_INDEX] = [UKN_TOKEN]
        # self.destination_vocab_list[UKN_TOKEN_INDEX] = [UKN_TOKEN]
        
        self.source_vocab_list += list(self.source_vocab)
        self.destination_vocab_list += list(self.destination_vocab)

        
        # self.source_vocab_list.append(' ')
        # self.destination_vocab_list.append(' ')
        # self.source_vocab_list.append('_')
        # self.destination_vocab_list.append('_')
        # self.source_vocab_list.append('<SOS>')
        # self.destination_vocab_list.append('<SOS>')
        # self.source_vocab_list.append('<EOS>')
        # self.destination_vocab_list.append('<EOS>')
        # self.source_vocab_list.append('<unk>')
        # self.destination_vocab_list.append('<unk>')

        logger.info(f"source_vocab len {len(list(self.source_vocab))}, destination vocab len {len(list(self.destination_vocab))}")
    
    def get_samplers(self, ):
        
        self.train_sampler = SubsetRandomSampler(self.train_inds)
        self.validation_sampler = SubsetRandomSampler(self.validate_inds)
        self.test_sampler = SubsetRandomSampler(self.test_inds)
        
        return self.train_sampler, self.validation_sampler, self.test_sampler
    


class TextData(Dataset):
    """Alphabets data loadder"""
    
    source_vocab = None
    destination_vocab = None
    
    def __init__(self, 
                 data = [],
                 source_vocab=[],
                 destination_vocab=[],
                 unknown_char=UKN_TOKEN,
                 transform=None,
                 max_len=0):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.transform = transform
        self.data = data
        self.source_vocab = source_vocab
        self.destination_vocab = destination_vocab
        self.unknown_char = unknown_char
        self.max_len = max_len
        
        TextData.source_vocab = self.source_vocab
        TextData.destination_vocab = self.destination_vocab
        
    # def __new__(cls, *args, **kwargs):
    #   # obj = super().__new__(cls, *args, **kwargs)
    #   obj = super(TextData, cls).__new__(cls, *args, **kwargs)
      
    #   if isinstance(obj, cls):
    #     obj.__init__(*args, **kwargs)
        
    #   cls.source_vocab = kwargs.get("source_vocab")
    #   cls.destination_vocab = kwargs.get("destination__vocab")
      
    #   return obj
      
    def __len__(self):
        return len(self.data) if self.data else 0

    def __add_mandatory_tokens__(self, vector, type_):
      
      if type_ == "source":
        v = vector + [EOS_TOKEN_INDEX]
        return v
      else:
        v = [SOS_TOKEN_INDEX] + vector + [EOS_TOKEN_INDEX]
        return v

    def __create_vector__(self, tokens, vocab):
      vect = torch.ones(self.max_len) * PAD_TOKEN_INDEX
      
      ## Adding starting position with <SOS>
      vect[0] = SOS_TOKEN_INDEX
      
      if '' in tokens:
        logger.exception(f"tokens : {tokens}")
        exit()
      for pos, token in enumerate(tokens):
        try:
          if (pos + 1) > len(tokens):
            raise IndexError
          vect[pos + 1] = vocab.index(token)
        except IndexError as e:
          vect[pos] = EOS_TOKEN_INDEX
        except ValueError as e:
          try:
            vect[pos + 1] = UKN_TOKEN_INDEX
          except IndexError as e:
            vect[pos] = EOS_TOKEN_INDEX
      
      vect[-1] = EOS_TOKEN_INDEX
      return vect
      
    
    def __end_token__(self):
      return [self.source_vocab.index('<EOS>')]

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x, y = self.data[idx]
        
        x_original = x
        y_original = y
        x, y = x.strip().split(' ')[:self.max_len], y.strip().split(' ')[:self.max_len]
        
        if self.source_vocab and self.destination_vocab:

            
            x = self.__create_vector__(x, self.source_vocab)
            y = self.__create_vector__(y, self.destination_vocab)

            
            # with open(DATA_PATH+"/vectors/data.en", "a") as f:
            #     f.write(" ".join([str(_) for _ in list(x.numpy())]) + "\n")
        
            # with open(DATA_PATH+"/vectors/data.te", "a") as f:
            #     f.write(" ".join([str(_) for _ in list(y.numpy())]) + "\n")
                    
        sample = x, y, x_original, y_original
        if self.transform:
            sample = self.transform(sample)

        return sample


def collate_fun(batch):
    source_list, dest_list = [], []

    try:
      for (_s, _d,x_original, y_original) in batch:
        source_list.append(_s)
        dest_list.append(_d)
          
      source_list = torch.stack(source_list)
      dest_list = torch.stack(dest_list)
    except ValueError as e:
      logger.exception(f"Origianl sentance :{x_original}, {y_original} Exception while converting into tensor : {e} source_list : {source_list} dest_list : {dest_list}")
    return (pad_sequence(source_list, padding_value=float(PAD_TOKEN_INDEX),batch_first=True).to(torch.long),
            pad_sequence(dest_list, padding_value=float(PAD_TOKEN_INDEX), batch_first=True).to(torch.long))
    
    
def get_data_generators():
    data_loader = LoadAndData()
    
    data_loader.run()
    data_loader.max_sent_len()
    data_loader.train_data()
    data_loader.build_vocab()
    data_loader.get_samplers()
    
    test_data_loader = TextData(data=data_loader.data, 
                                source_vocab=data_loader.source_vocab_list, 
                                destination_vocab=data_loader.destination_vocab_list,
                                max_len=data_loader.max_sent_len())
    validate_data_loader = TextData(data=data_loader.data, 
                                source_vocab=data_loader.source_vocab_list, 
                                destination_vocab=data_loader.destination_vocab_list,
                                max_len=data_loader.max_sent_len())
    train_data_loader = TextData(data=data_loader.data, 
                                source_vocab=data_loader.source_vocab_list, 
                                destination_vocab=data_loader.destination_vocab_list,
                                max_len=data_loader.max_sent_len())
    
        
    train_params = {'batch_size': BATCH_SIZE,
            'shuffle': False,
            'num_workers': 6,
            'collate_fn': collate_fun,
            'drop_last': True,
            'sampler': data_loader.train_sampler}


    test_params = {'batch_size': BATCH_SIZE,
            'shuffle': False,
            'num_workers': 6,
            'collate_fn': collate_fun,
            'drop_last': True,
            'sampler': data_loader.test_sampler}

    validation_params = {'batch_size': BATCH_SIZE,
            'shuffle': False,
            'num_workers': 6,
            'collate_fn': collate_fun,
            'drop_last': True,
            'sampler': data_loader.validation_sampler}
    
    
    training_generator = DataLoader(train_data_loader, **train_params,)
    test_data_generator = DataLoader(test_data_loader, **test_params)
    validate_data_generator = DataLoader(validate_data_loader, **validation_params)
    
    logger.info(f"validate_data_generator len : {len(validate_data_generator)}")
    
    return training_generator, test_data_generator, validate_data_generator
