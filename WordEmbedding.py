import os


class FeatureBuilder:
    def __init__(self, input_path = "./CONLL_train.pos-chunk-name", train_mode = True):
        self.out_path = input_path[input_path.rfind("/")+1: input_path.rfind(".")] + ".feature"
        self.in_file = open(input_path, 'r')
        self.out_file = open(self.out_path, 'w')
        self.train_mode = train_mode

        self.write_count = 0

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

    def close_file(self):
        self.in_file.close()
        self.out_file.close()

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
                #

            feature = list(f for i, f in enumerate(all_feature) if enable_list[i])

            feature.append(self.add_word_embedding(token, self.threshold))

            feature_size = len(feature)

            if token == "-DOCSTART-":
                feature = ("-X-" for i in range(feature_size))

            if self.train_mode:
                self.append_feature(token, feature, tag)
            else:
                self.append_feature(token, feature, None)

        self.out_file.write("\n")

    def add_word_embedding(self, word, threshold):
        ret = [0 for i in range(50)]
        return ret

    def run(self):
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

