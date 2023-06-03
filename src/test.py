result = "<SOS> a man and woman sit on a bench . <EOS>"
result = result.replace("<SOS>", "")
result = result.replace("<EOS>", "")
result = result.replace(".", "")

print(result)