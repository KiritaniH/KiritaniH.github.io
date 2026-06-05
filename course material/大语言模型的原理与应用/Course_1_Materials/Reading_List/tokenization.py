############################################################
# Tokenization

# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py#L23
GPT2_TOKENIZER_REGEX = \
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def tokenization():
    text("This unit was inspired by Andrej Karpathy's video on tokenization; check it out! "), youtube_link("https://www.youtube.com/watch?v=zduSFxRajkE")

    intro_to_tokenization()
    tokenization_examples()
    character_tokenizer()
    byte_tokenizer()
    word_tokenizer()
    bpe_tokenizer()

    text("## Summary")
    text("- Tokenizer: strings <-> tokens (indices)")
    text("- Character-based, byte-based, word-based tokenization highly suboptimal")
    text("- BPE is an effective heuristic that looks at corpus statistics")
    text("- Tokenization is a necessary evil, maybe one day we'll just do it from bytes...")

@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: dict[int, bytes]     # index -> bytes
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index



class CharacterTokenizer(Tokenizer):
    """Represent a string as a sequence of Unicode code points."""
    def encode(self, string: str) -> list[int]:
        return list(map(ord, string))

    def decode(self, indices: list[int]) -> str:
        return "".join(map(chr, indices))


class ByteTokenizer(Tokenizer):
    """Represent a string as a sequence of bytes."""
    def encode(self, string: str) -> list[int]:
        string_bytes = string.encode("utf-8")  # @inspect string_bytes
        indices = list(map(int, string_bytes))  # @inspect indices
        return indices

    def decode(self, indices: list[int]) -> str:
        string_bytes = bytes(indices)  # @inspect string_bytes
        string = string_bytes.decode("utf-8")  # @inspect string
        return string


def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:  # @inspect indices, @inspect pair, @inspect new_index
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []  # @inspect new_indices
    i = 0  # @inspect i
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8")))  # @inspect indices
        # Note: this is a very slow implementation
        for pair, new_index in self.params.merges.items():  # @inspect pair, @inspect new_index
            indices = merge(indices, pair, new_index)
        return indices

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))  # @inspect bytes_list
        string = b"".join(bytes_list).decode("utf-8")  # @inspect string
        return string


def get_compression_ratio(string: str, indices: list[int]) -> float:
    """Given `string` that has been tokenized into `indices`, ."""
    num_bytes = len(bytes(string, encoding="utf-8"))  # @inspect num_bytes
    num_tokens = len(indices)                       # @inspect num_tokens
    return num_bytes / num_tokens


def get_gpt2_tokenizer():
    # Code: https://github.com/openai/tiktoken
    # You can use cl100k_base for the gpt3.5-turbo or gpt4 tokenizer
    return tiktoken.get_encoding("gpt2")


def intro_to_tokenization():
    text("Raw text is generally represented as Unicode strings.")
    string = "Hello, 🌍! 你好!"

    text("A language model places a probability distribution over sequences of tokens (usually represented by integer indices).")
    indices = [15496, 11, 995, 0]

    text("So we need a procedure that *encodes* strings into tokens.")
    text("We also need a procedure that *decodes* tokens back into strings.")
    text("A "), link(Tokenizer), text(" is a class that implements the encode and decode methods.")
    text("The **vocabulary size** is number of possible tokens (integers).")


def tokenization_examples():
    text("To get a feel for how tokenizers work, play with this "), link(title="interactive site", url="https://tiktokenizer.vercel.app/?encoder=gpt2")

    text("## Observations")
    text("- A word and its preceding space are part of the same token (e.g., \" world\").")
    text("- A word at the beginning and in the middle are represented differently (e.g., \"hello hello\").")
    text("- Numbers are tokenized into every few digits.")

    text("Here's the GPT-2 tokenizer from OpenAI (tiktoken) in action.")
    tokenizer = get_gpt2_tokenizer()
    string = "Hello, 🌍! 你好!"  # @inspect string

    text("Check that encode() and decode() roundtrip:")
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio


def character_tokenizer():
    text("## Character-based tokenization")

    text("A Unicode string is a sequence of Unicode characters.")
    text("Each character can be converted into a code point (integer) via `ord`.")
    assert ord("a") == 97
    assert ord("🌍") == 127757
    text("It can be converted back via `chr`.")
    assert chr(97) == "a"
    assert chr(127757) == "🌍"

    text("Now let's build a `Tokenizer` and make sure it round-trips:")
    tokenizer = CharacterTokenizer()
    string = "Hello, 🌍! 你好!"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string

    text("There are approximately 150K Unicode characters. "), link(title="[Wikipedia]", url="https://en.wikipedia.org/wiki/List_of_Unicode_characters")
    vocabulary_size = max(indices) + 1  # This is a lower bound @inspect vocabulary_size
    text("Problem 1: this is a very large vocabulary.")
    text("Problem 2: many characters are quite rare (e.g., 🌍), which is inefficient use of the vocabulary.")
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio


def byte_tokenizer():
    text("## Byte-based tokenization")

    text("Unicode strings can be represented as a sequence of bytes, which can be represented by integers between 0 and 255.")
    text("The most common Unicode encoding is "), link(title="UTF-8", url="https://en.wikipedia.org/wiki/UTF-8")

    text("Some Unicode characters are represented by one byte:")
    assert bytes("a", encoding="utf-8") == b"a"
    text("Others take multiple bytes:")
    assert bytes("🌍", encoding="utf-8") == b"\xf0\x9f\x8c\x8d"

    text("Now let's build a `Tokenizer` and make sure it round-trips:")
    tokenizer = ByteTokenizer()
    string = "Hello, 🌍! 你好!"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string

    text("The vocabulary is nice and small: a byte can represent 256 values.")
    vocabulary_size = 256  # @inspect vocabulary_size
    text("What about the compression rate?")
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio
    assert compression_ratio == 1
    text("The compression ratio is terrible, which means the sequences will be too long.")
    text("Given that the context length of a Transformer is limited (since attention is quadratic), this is not looking great...")


def word_tokenizer():
    text("## Word-based tokenization")

    text("Another approach (closer to what was done classically in NLP) is to split strings into words.")
    string = "I'll say supercalifragilisticexpialidocious!"

    segments = regex.findall(r"\w+|.", string)  # @inspect segments
    text("This regular expression keeps all alphanumeric characters together (words).")

    text("Here is a fancier version:")
    pattern = GPT2_TOKENIZER_REGEX  # @inspect pattern
    segments = regex.findall(pattern, string)  # @inspect segments

    text("To turn this into a `Tokenizer`, we need to map these segments into integers.")
    text("Then, we can build a mapping from each segment into an integer.")

    text("But there are problems:")
    text("- The number of words is huge (like for Unicode characters).")
    text("- Many words are rare and the model won't learn much about them.")
    text("- This doesn't obviously provide a fixed vocabulary size.")

    text("New words we haven't seen during training get a special UNK token, which is ugly and can mess up perplexity calculations.")

    vocabulary_size = "Number of distinct segments in the training data"
    compression_ratio = get_compression_ratio(string, segments)  # @inspect compression_ratio


def bpe_tokenizer():
    text("## Byte Pair Encoding (BPE)")
    link(title="[Wikipedia]", url="https://en.wikipedia.org/wiki/Byte_pair_encoding")
    text("The BPE algorithm was introduced by Philip Gage in 1994 for data compression. "), article_link("http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM")
    text("It was adapted to NLP for neural machine translation. "), link(sennrich_2016)
    text("(Previously, papers had been using word-based tokenization.)")
    text("BPE was then used by GPT-2. "), link(gpt2)

    text("Basic idea: *train* the tokenizer on raw text to automatically determine the vocabulary.")
    text("Intuition: common sequences of characters are represented by a single token, rare sequences are represented by many tokens.")

    text("The GPT-2 paper used word-based tokenization to break up the text into inital segments and run the original BPE algorithm on each segment.")
    text("Sketch: start with each byte as a token, and successively merge the most common pair of adjacent tokens.")

    text("## Training the tokenizer")
    string = "the cat in the hat"  # @inspect string
    params = train_bpe(string, num_merges=3)

    text("## Using the tokenizer")
    text("Now, given a new text, we can encode it.")
    tokenizer = BPETokenizer(params)
    string = "the quick brown fox"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string

    text("In Assignment 1, you will go beyond this in the following ways:")
    text("- encode() currently loops over all merges. Only loop over merges that matter.")
    text("- Detect and preserve special tokens (e.g., <|endoftext|>).")
    text("- Use pre-tokenization (e.g., the GPT-2 tokenizer regex).")
    text("- Try to make the implementation as fast as possible.")


def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:  # @inspect string, @inspect num_merges
    text("Start with the list of bytes of `string`.")
    indices = list(map(int, string.encode("utf-8")))  # @inspect indices
    merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes

    for i in range(num_merges):
        text("Count the number of occurrences of each pair of tokens")
        counts = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):  # For each adjacent pair
            counts[(index1, index2)] += 1  # @inspect counts

        text("Find the most common pair.")
        pair = max(counts, key=counts.get)  # @inspect pair
        index1, index2 = pair

        text("Merge that pair.")
        new_index = 256 + i  # @inspect new_index
        merges[pair] = new_index  # @inspect merges
        vocab[new_index] = vocab[index1] + vocab[index2]  # @inspect vocab
        indices = merge(indices, pair, new_index)  # @inspect indices

    return BPETokenizerParams(vocab=vocab, merges=merges)


if __name__ == "__main__":
    main()