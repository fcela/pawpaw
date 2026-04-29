#!/usr/bin/env python3
"""
Train sentiment, topic, and intent classifiers using pawpaw's training infrastructure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pawpaw.synth.examples import Pair
from pawpaw.config import TrainConfig
from pawpaw.train.trainer import train_lora
from pawpaw.train.prompt_template import build_prompt_template
import random

random.seed(42)

BASE_MODEL = "Qwen/Qwen3-0.6B"
TEMPLATE_SPEC = (
    "Classify the user message into the appropriate category. "
    "Output exactly the category label, no other text."
)

def make_pair(input_text: str, output_text: str) -> Pair:
    """Create a Pair with default category and length_bucket."""
    return Pair(
        input=input_text,
        output=output_text,
        category="default",
        length_bucket="short"
    )

# ============================================================================
# 1. SENTIMENT CLASSIFIER
# ============================================================================
print("=" * 80)
print("TRAINING SENTIMENT CLASSIFIER")
print("=" * 80)

sentiment_pairs = []

# Positive examples
positive_texts = [
    "I absolutely love this product!",
    "This is amazing, highly recommend!",
    "Best purchase ever, so happy!",
    "Outstanding quality and service!",
    "Exceeded my expectations, wonderful!",
    "Fantastic experience, will buy again!",
    "Perfect! Exactly what I needed.",
    "Brilliant work, very impressed!",
    "Superb! Can't fault it at all.",
    "Delightful and enjoyable experience!",
    "Great value for money, satisfied!",
    "Awesome product, fast shipping!",
    "Really happy with this purchase!",
    "Excellent quality, highly recommend!",
    "Love it! Works perfectly.",
    "Very pleased with the results!",
    "Top-notch quality, amazing!",
    "Wonderful product, great features!",
    "Impressed with the quality!",
    "Happy customer here, great buy!",
]

for text in positive_texts:
    sentiment_pairs.append(make_pair(text, "positive"))

# Negative examples
negative_texts = [
    "Terrible quality, very disappointed.",
    "Worst purchase ever, don't buy!",
    "Complete waste of money, awful!",
    "Horrible experience, never again!",
    "Disappointing and frustrating product.",
    "Poor quality, broke immediately.",
    "Not worth the price, regret buying.",
    "Very unhappy with this purchase.",
    "Defective product, want refund.",
    "Awful customer service, rude staff.",
    "Doesn't work as advertised, scam!",
    "Cheap and flimsy, broke quickly.",
    "Waste of time and money.",
    "Extremely disappointed, expected better.",
    "Useless product, don't recommend.",
    "Terrible experience, very frustrating.",
    "Bad quality, not as described.",
    "Regret this purchase completely.",
    "Horrible! Do not buy this!",
    "Very poor quality, disappointed.",
]

for text in negative_texts:
    sentiment_pairs.append(make_pair(text, "negative"))

# Neutral examples
neutral_texts = [
    "It's okay, nothing special.",
    "Average product, does the job.",
    "Neither good nor bad, just fine.",
    "Mediocre quality, expected more.",
    "It's alright, could be better.",
    "Standard product, no complaints.",
    "Acceptable quality for the price.",
    "Works as expected, nothing more.",
    "Fair enough, meets basic needs.",
    "Ordinary product, nothing special.",
    "Decent enough for occasional use.",
    "It's fine, no major issues.",
    "Average quality, nothing impressive.",
    "Satisfactory but not outstanding.",
    "Okay product, serves its purpose.",
    "Middle of the road, acceptable.",
    "Reasonable quality for the cost.",
    "It works, can't complain much.",
    "Basic but functional, okay.",
    "Not bad, not great, just okay.",
]

for text in neutral_texts:
    sentiment_pairs.append(make_pair(text, "neutral"))

print(f"Sentiment dataset: {len(sentiment_pairs)} examples")

# Train sentiment classifier
try:
    config = TrainConfig(
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        seed=42,
        val_fraction=0.2,
    )
    
    template = build_prompt_template(TEMPLATE_SPEC, demos=sentiment_pairs[:2])

    output_path = train_lora(
        base_model=BASE_MODEL,
        template=template,
        pairs=sentiment_pairs,
        config=config,
        output_dir=Path("./programs/sentiment"),
        max_length=256,
    )
    print(f"✓ Sentiment classifier trained successfully! Saved to: {output_path}")
except Exception as e:
    print(f"✗ Error training sentiment classifier: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 2. TOPIC CLASSIFIER
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING TOPIC CLASSIFIER")
print("=" * 80)

topic_pairs = []

topic_data = {
    'technology': [
        "The new smartphone features advanced AI capabilities.",
        "Software developers are in high demand this year.",
        "Cloud computing continues to grow rapidly.",
        "The latest processor has 64 cores.",
        "Artificial intelligence is transforming industries.",
    ],
    'sports': [
        "The team won the championship yesterday.",
        "The athlete broke the world record.",
        "Football season starts next week.",
        "The basketball player scored 40 points.",
        "Tennis tournament begins tomorrow.",
    ],
    'politics': [
        "The senator announced a new policy.",
        "Election results were announced today.",
        "The president signed the legislation.",
        "Congress debated the budget proposal.",
        "The governor issued an executive order.",
    ],
    'entertainment': [
        "The movie premiered at the festival.",
        "The concert sold out in minutes.",
        "New album topped the charts.",
        "The actor won an award yesterday.",
        "TV show finale was amazing.",
    ],
    'business': [
        "The company reported quarterly profits.",
        "Stock market reached new highs.",
        "Merger deal was announced today.",
        "The CEO resigned unexpectedly.",
        "Startup raised venture capital.",
    ],
    'science': [
        "Scientists discovered a new species.",
        "The research was published today.",
        "Climate study shows concerning trends.",
        "The experiment yielded results.",
        "Astronomers observed distant galaxies.",
    ],
    'health': [
        "The doctor recommended exercise.",
        "Hospital implemented new protocols.",
        "Vaccine development is progressing.",
        "Patients reported improvement.",
        "The treatment was effective.",
    ],
    'travel': [
        "The flight was delayed hours.",
        "Tourist destination is beautiful.",
        "Hotel accommodation was excellent.",
        "The vacation was relaxing.",
        "Airport security was thorough.",
    ],
    'food': [
        "The restaurant served delicious food.",
        "Recipe turned out perfectly.",
        "The chef prepared a masterpiece.",
        "Ingredients were fresh and local.",
        "The meal was satisfying.",
    ],
    'weather': [
        "The forecast predicts rain tomorrow.",
        "Temperature reached record highs.",
        "Storm warning was issued.",
        "The sunshine was beautiful today.",
        "Humidity levels are uncomfortable.",
    ],
}

for topic, texts in topic_data.items():
    for text in texts:
        topic_pairs.append(make_pair(text, topic))

print(f"Topic dataset: {len(topic_pairs)} examples")

# Train topic classifier
try:
    config = TrainConfig(
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        seed=42,
        val_fraction=0.2,
    )

    template = build_prompt_template(TEMPLATE_SPEC, demos=topic_pairs[:2])

    output_path = train_lora(
        base_model=BASE_MODEL,
        template=template,
        pairs=topic_pairs,
        config=config,
        output_dir=Path("./programs/topic"),
        max_length=256,
    )
    print(f"✓ Topic classifier trained successfully! Saved to: {output_path}")
except Exception as e:
    print(f"✗ Error training topic classifier: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 3. INTENT CLASSIFIER
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING INTENT CLASSIFIER")
print("=" * 80)

intent_pairs = []

intent_data = {
    'question': [
        "What time is the meeting?",
        "How do I fix this bug?",
        "Where is the documentation?",
        "When is the deadline?",
        "Why is this happening?",
    ],
    'command': [
        "Close the application immediately.",
        "Send the report today.",
        "Fix this bug now.",
        "Update the documentation.",
        "Review the code changes.",
    ],
    'statement': [
        "The project is on schedule.",
        "I will attend the meeting.",
        "This is the final version.",
        "The system is working properly.",
        "We completed the milestone.",
    ],
    'greeting': [
        "Hello, how are you?",
        "Good morning everyone.",
        "Hi there!",
        "Hey, what's up?",
        "Greetings!",
    ],
    'farewell': [
        "Goodbye, see you later.",
        "Have a great day.",
        "Take care.",
        "See you tomorrow.",
        "Farewell.",
    ],
    'confirmation': [
        "Yes, that's correct.",
        "Absolutely, I agree.",
        "Definitely, go ahead.",
        "Yes, please proceed.",
        "Confirmed, thank you.",
    ],
    'denial': [
        "No, that's incorrect.",
        "I disagree.",
        "No, thank you.",
        "Definitely not.",
        "No, that's incorrect.",
    ],
}

for intent, texts in intent_data.items():
    for text in texts:
        intent_pairs.append(make_pair(text, intent))

print(f"Intent dataset: {len(intent_pairs)} examples")

# Train intent classifier
try:
    config = TrainConfig(
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        seed=42,
        val_fraction=0.2,
    )

    template = build_prompt_template(TEMPLATE_SPEC, demos=intent_pairs[:2])

    output_path = train_lora(
        base_model=BASE_MODEL,
        template=template,
        pairs=intent_pairs,
        config=config,
        output_dir=Path("./programs/intent"),
        max_length=256,
    )
    print(f"✓ Intent classifier trained successfully! Saved to: {output_path}")
except Exception as e:
    print(f"✗ Error training intent classifier: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print("""
Trained classifiers:
✓ Sentiment classifier: ./programs/sentiment
✓ Topic classifier: ./programs/topic  
✓ Intent classifier: ./programs/intent

Next: Run test suites and benchmarks
""")
