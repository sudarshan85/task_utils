#!/usr/bin/env python
import unicodedata
import re
import spacy
sci = spacy.load('en_core_sci_sm', disable=['parser', 'tagger', 'ner'])

def tokenize(text, stop_words=set()):
  text = text.lower()
  doc = sci(text)
  tokens = []
  for token in doc:
    if token.is_punct or token.is_space or token.text in stop_words:
      continue
    tokens.append(token.lemma_)

  return tokens

def convert_to_unicode(text):
  if isinstance(text, str):
    return text
  elif isinstance(text, bytes):
    return text.decode('utf-8', 'ignore')
  else:
    raise ValueError(f"Unsupported string type: {type(text)}")

def is_control(char):
  if char == '\t' or char == '\n' or char == '\r':
    return False
  cat = unicodedata.category(char)
  if cat in ('Cc', 'Cf'):
    return True

  return False

def is_whitespace(char):
  if char == '\t' or char == '\n' or char == '\r':
    return True
  cat = unicodedata.category(char)
  if cat == 'Zs':
    return True

  return False

def is_punc(char):
  cp = ord(char)
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith('P'):
    return True

  return False

def whitespace_tokenize(text):
  return text.split() if text.strip() else []

def clean_text(text, remove_punc=False):
  output = []
  for char in text:
    cp = ord(char)
    if cp == 0 or cp == 0xfffd or is_control(char):
      continue
    if remove_punc and is_punc(char):
      # output.append(' ')
      continue
    if is_whitespace(char):
      output.append(' ')
    else:
      output.append(char)
  output = re.sub(' +', ' ', ''.join(output).lower())
  return ''.join(output)

class Tokenizer(object):
  def __init__(self, do_lower=True, remove_punc=False):
    self.do_lower = do_lower
    self.remove_punc = remove_punc

  def tokenize(self, text):
    text = convert_to_unicode(text)
    text = clean_text(text, self.remove_punc)

    split_tokens = []
    for token in whitespace_tokenize(text):
      if self.do_lower:
        token = token.lower()
      token = self._strip_accents(token)
      split_tokens.extend(self._split_on_punc(token))

    return whitespace_tokenize(' '.join(split_tokens))

  def _strip_accents(self, token):
    token = unicodedata.normalize('NFD', token)
    output = []
    for char in token:
      cat = unicodedata.category(char)
      if cat == 'Mn':
        continue
      output.append(char)
    return ''.join(output)

  def _split_on_punc(self, token):
    chars = list(token)
    i = 0
    start_new_token = True
    output = []
    while i < len(chars):
      char = chars[i]
      if is_punc(char):
        output.append([char])
        start_new_token = True
      else:
        if start_new_token:
          output.append([])
        start_new_token = False
        output[-1].append(char)
      i += 1

    return [''.join(x) for x in output]
