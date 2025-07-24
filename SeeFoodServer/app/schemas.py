from pydantic import BaseModel


class Base64Image(BaseModel):
    image: str