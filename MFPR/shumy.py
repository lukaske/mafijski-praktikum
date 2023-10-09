
replacables = {
    "ˇc": "č",
    "ˇs": "š",
    "ˇz": "ž",
    "ˇC": "Č",
    "Cˇ": "Č",
    "ˇZ": "Ž",
        "Zˇ": "Ž",
    "ˇS": "Š",
    "Sˇ": "Š",

    }
replacables2 = {
    "·c": "č",
    "·s": "š",
    "·z": "ž",
    "·C": "Č",
    "·Z": "Ž",
    "·S": "Š",
    "\n": " "
    }


with open("dodajsumnike", encoding="UTF-8") as file:
    text = file.readlines()
    text = "".join(text)
    for key, value in replacables.items():
        text = text.replace(key, value)
    for key, value in replacables2.items():
        text = text.replace(key, value)

    print(text)