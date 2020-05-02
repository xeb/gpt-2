#!/usr/bin/env python

import re
import os

def sanitize(text):
    # Sanitize text for actual use
    print(f"-------\n... sanitizer, orig: {text}")
    ptext = text.split('<EOM>')
    result = "".join(ptext[:4])

    names = os.environ["DIXIE_NAMES"].split(',')
    for name in names:
        for i in [0,1]:
            result = result.replace(f"{i}|{name}|","")
            result = result.replace(f"{i}|+{name}|","")

    result = result.replace(f"|{name}|","")
    result = result.replace(f"|+{name}|","")


    prepurge = ["|1|","|0|", ">1|", ">0|", "0|", "1|","--", "&"]
    for p in prepurge:
        result = result.replace(p, "")

    purgables = ["<EOM>|"," 2", "<EOM", "<EO", "<E", "(", ")", " 0 ", " 1 ","<",">", "|","https","http","<EOM>","+","❤️","☕️","=","😬"]
    for p in purgables:
        result = result.replace(p, " ")

    result = re.sub(' +', ' ', result)
    print(f"-------\n... sanitizer, result: {result}")
    
    for p in purgables:
        result = result.replace(p, " ")

    for p in ["0 ", "1 "]:
        result = result.replace(p, "")

    result = re.sub(' +', ' ', result)

    replacements = {
            "COE": "C. O. E.",
            "ADG": "A. D. G.",
            "SEA": "S. E. A.",
    }

    for r in replacements.keys():
        result = result.replace(r, replacements[r])

    return result

if __name__ == "__main__":
    t = """1|yahia|I don't have access yet <EOM>0|yahia|I don't mind sending it to you, it will be public <EOM>1|yahia|Thank You! <EOM>0|yahia|TOM! Did we get a resourcing matrix? I was under the impression that Christine did the initial selection! And then Neha did a supplemental interview <EOM> 1|yahia|That’s’right. Neha got a 1 <EOM> 0|yahia|oh that'"""
    print(f"Sanitizing {t}")
    result = sanitize(t)
    print(f"Result:\n{result}")