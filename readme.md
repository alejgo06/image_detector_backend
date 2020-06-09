# image detector service

This api receives an image return the coordinate where a persona, animal or object is located

Each version is a improve from previous versions
# versions:

- V1. {people, dogs,cats}
- V2. {people, dogs, cats, other}
- V3. {people, dogs, cats, other, objects}

# run 

uvicorn backend:app --reload --port 8001

http://127.0.0.1:8001/docs#/default

