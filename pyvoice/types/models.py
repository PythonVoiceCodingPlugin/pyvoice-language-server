from pydantic import BaseModel


class Model(BaseModel):
    """the base model we are going to use for all our models.


    By default it is going to be immutable"""

    class Config:
        faux_immutability = True
