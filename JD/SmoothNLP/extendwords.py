
old_words = set()
f = open("../CommonData/dict")
while True:
    lines = f.readline()
    if not lines:
        break
    old_words.add(lines.split("\t")[0])

new_words=""
f = open("./output3000w.txt")
new_words = f.readline()

new_words = new_words.replace("'","")
new_words = new_words.replace(" ","")
new_words =  new_words[2:-2].split(",")

res = []
for tstr in new_words:
    if tstr not in old_words:
        res.append(tstr)

print(res)
print(len(res))
