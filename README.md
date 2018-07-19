# Card_OCR

# Firstly
create folders named processed cardfinal
# Secondly
call

python cardfinder.py [path_to_image] [auto/manual]

if auto:
it will automatically find and crop the card out
if manual:
it will prompt 4 inputs of the quad points circumsizing the card

the result will be showing in the cardfinal
# Thirdly
call

python recognizer.py [path_to_image]

You should load images in cardfinal folder

the result is in stdout
