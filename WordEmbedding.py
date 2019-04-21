import os
import scipy as sp
import numpy as np

class FeatureBuilder:
    def __init__(self, input_path = "./CONLL_train.pos-chunk-name", train_mode = True):
        self.out_path = input_path[input_path.rfind("/")+1: input_path.rfind(".")] + ".feature"
        self.word_embedding_filepath = "./glove.6B/glove.6B.50d.txt"
        self.in_file = open(input_path, 'r')
        self.out_file = open(self.out_path, 'w')
        self.wbf = open(self.word_embedding_filepath, 'r')
        self.wb = {}
        self.train_mode = train_mode
        self.write_count = 0
        self.count_embed_word = 0
        self.count_word = 0
# TODO: try thresholds.
        self.threshold = 0.3
        self.trshd_pos = np.zeros(50)
        self.trshd_neg = np.zeros(50)

    @staticmethod
    def exec_line(line):
        for i in range(0, 32):
            line = line.replace(chr(i), " ")
        line = line.strip().split(" ")
        return tuple(i for i in line)

    def append_feature(self, token, feature, tag):
        ret = token
        for f in feature:
            ret = ret + "\t" + "feature=" + str(f)
        if self.train_mode:
            ret = ret + "\t" + str(tag)
        ret = ret + "\n"
        self.out_file.write(ret)

    def get_word_embed(self):
        lines = self.wbf.readlines()
        count_pos = np.zeros(50)
        count_neg = np.zeros(50)

        for line in lines:
            line = line.split(" ")
            token = line[0]
            feature = np.asarray([float(i) for i in line[1:]])
            for i, val in enumerate(feature):
                if val > 0:
                    count_pos[i] += 1
                    self.trshd_pos[i] += val
                else:
                    count_neg[i] += 1
                    self.trshd_neg[i] += val
            self.wb[token] = feature

        self.trshd_pos = self.trshd_pos / count_pos
        self.trshd_neg = self.trshd_neg / count_neg
        print(count_pos)
        print(count_neg)


    def close_file(self):
        self.in_file.close()
        self.out_file.close()
        self.wbf.close()

    def exec_sentence(self, sentence):
        length = len(sentence)
        enable_list = None
        for i, data in enumerate(sentence):
            if self.train_mode:
                token, pos, bio, tag = data[0], data[1], data[2], data[3]
            else:
                token, pos, bio = data[0], data[1], data[2]

            if i > 0:
                pre = sentence[i - 1]
            else:
                pre = ("None", "start", 0, 0)
            if i < length - 1:
                post = sentence[i + 1]
            else:
                post = ("None", "end", 0, 0)

            token_lower = token.lower()

            prev_tag = "@@"
            if self.train_mode and i > 0:
                prev_tag = sentence[i-1][3]

            upper_char_count = 0

            for ch in token:
                if ch.isupper():
                    upper_char_count += 1

            # count_non_alphanum = len(token)
            # count_non_alpha = len(token)
            # for ch in token_lower:
            #     if ch.isalpha():
            #         count_non_alpha -= 1
            #     if ch.isalnum():
            #         count_non_alphanum -= 1

            all_feature = [pre[1], post[1], pre[2], post[2],
                           len(token), token, pre[0], post[0], pos, bio, token.islower(), token.find("-"), prev_tag,
                           token_lower, pre[0].lower(), upper_char_count, upper_char_count / len(token),
                           token_lower.find("bach")]

            if not enable_list:
                enable_list = [1 for i in range(len(all_feature))]
                # disable some feature
                for i in range(4):
                    enable_list[i] = 0

            feature = list(f for i, f in enumerate(all_feature) if enable_list[i])

            feature.append(self.add_word_embedding_bin(token))

            feature_size = len(feature)

            if token == "-DOCSTART-":
                feature = ("-X-" for i in range(feature_size))

            if self.train_mode:
                self.append_feature(token, feature, tag)
            else:
                self.append_feature(token, feature, None)

        self.out_file.write("\n")

    def add_word_embedding_bin_mean(self, word):
        self.count_word += 1
        if word in self.wb:
            self.count_embed_word += 1
            ret = [(i > self.trshd_pos[index])*1 + (i < -self.trshd_neg[index]) for index, i in enumerate(self.wb[word])]
        else:
            ret = [0 for i in range(50)]
        return ret

    def add_word_embedding_bin(self, word):
        self.count_word += 1
        if word in self.wb:
            self.count_embed_word += 1
            ret = [(i > self.threshold)*1 + (i < -self.threshold)*(-1) for i in self.wb[word]]
        else:
            ret = [0 for i in range(50)]
        return ret

    def run(self):
        self.get_word_embed()
        sentence = []
        count = 0
        count_line = 0
        while True:
            line = self.in_file.readline()
            if not line:
                break
            count_line += 1
            data = self.exec_line(line)
            sentence.append(data)
            if len(data) == 1:
                count += 1
                sentence.pop()
                self.exec_sentence(sentence)
                sentence = []

        print("Finished.")
        print("Output:", self.out_path)
        self.close_file()
        print("There is ", count_line, "lines in training file.")
        print("There is ", count, "sentences in training file.")
        print("There is ", self.count_word, "word in training file.")
        print("There is ", self.count_embed_word, "word in glove6B.50d.txt.")


if __name__ == '__main__':
    builder = FeatureBuilder(train_mode=True)
    builder.run()

    if not (os.path.exists("MEtrain.class") and os.path.exists("MEtag.class")):
        os.system("javac -cp ./maxent-3.0.0.jar:trove.jar ./*.java")

    model_name = "MEmodel.bin.gz"
    dev_name = "CONLL_dev.pos-chunk"
    dev_feature = "CONLL_dev.feature"
    dev_out = "response.name"

    test_name = "CONLL_test.pos-chunk"
    test_feature = "CONLL_test.feature"
    test_out = "CONLL_test.name"

    os.system("java -cp .:./maxent-3.0.0.jar:trove.jar MEtrain " + builder.out_path + " " + model_name)

    builder = FeatureBuilder(input_path=dev_name, train_mode=False)
    builder.run()

    os.system("java -cp .:./maxent-3.0.0.jar:trove.jar MEtag " + dev_feature + " " + model_name + " " + dev_out)
    os.system("python3 score.name.py")

    builder_test = FeatureBuilder(input_path=test_name, train_mode=False)
    builder_test.run()
    os.system("java -cp .:./maxent-3.0.0.jar:trove.jar MEtag " + test_feature + " " + model_name + " " + test_out)

