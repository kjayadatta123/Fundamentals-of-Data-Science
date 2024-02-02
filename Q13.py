customer_reviews = [
    "The product is excellent.",
    "I am satisfied with my purchase.",
    "The quality of the product is not as expected.",
    "Great value for the price.",
]

word_frequency = {}

for review in customer_reviews:
    words = review.lower().split()
    for word in words:
        word = word.strip('.,!?()[]{}:;"\'')
        if word in word_frequency:
            word_frequency[word] += 1
        else:
            word_frequency[word] = 1

print("Word Frequency Distribution:")
for word, frequency in word_frequency.items():
    print(f"{word}: {frequency} times")
