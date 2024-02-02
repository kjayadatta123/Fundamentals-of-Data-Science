import re
from collections import Counter
import matplotlib.pyplot as plt

text_data = [
    "The product does not work properly. Very buggy!",
    "I really like the design and look of this product. Stylish!",
    "Customer service was slow to respond but eventually resolved my issue."
]

all_text = " ".join(text_data)
cleaned_text = re.sub(r'[^\w\s]', '', all_text).lower()
words = cleaned_text.split()
word_counts = Counter(words)

N = int(input('Enter N: '))
top_words = word_counts.most_common(N)
top_words_txt = ", ".join([w[0] for w in top_words])
top_freqs_txt = ", ".join([str(w[1]) for w in top_words])

print(f"Top {N} words: {top_words_txt}")
print(f"Frequencies: {top_freqs_txt}")

plt.bar([w[0] for w in top_words], [w[1] for w in top_words])
plt.show()
