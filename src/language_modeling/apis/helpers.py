def make_error_object(msg: str) -> dict:
    """
    Helper method for generating the JSON error in case one of the APIs errors out.
    It is used to make the error objects unified
    :param msg: The message to return in the "message" field
    :return: Dict representing the error object
    """
    return {"status": "error", "message": msg}
