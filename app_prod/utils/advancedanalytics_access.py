# Imports
import requests


# Exceptions
class PasswordGetFailure(Exception):
    """
    Usage:
        Raised when getting password fails.
    """
    pass


# AAAccess Class
class AAAccess:
    """
    Usage:
        ps = AAAccess(list_id)
            ...
    @param list_id <str>: Password list identifier
    """

    def __init__(self, lid=915):
        self.uri = 'https://passwordstate.santos.com/api/'
        self.key = '2fc63e749d8a3b598b3256f3c4eda73d'
        self.lid = str(lid)

    def __enter__(self):
        return self

    def __repr__(self):
        return 'passwordstate.PasswordState(%r)' % (self.lid)

    def __str__(self):
        return 'PasswordState: [LID=%s]' % (self.lid)

    def _get_title_(self, system, user):
        return '"' + system + '_' + user + '"'

    def _get_pwd_(self, system, user):
        url = self.uri + 'searchpasswords/' + self.lid
        title = self._get_title_(system, user)
        params = dict(title=title, apikey=self.key)
        response = None
        password = None
        password_id = None
        try:
            response = requests.get(url, params=params).json()
            password = response[0]['Password']
            password_id = response[0]['PasswordID']
        except:
            raise PasswordGetFailure(response)
        return dict(password=password, password_id=password_id)

    def get_pwd(self, system, user):
        return self._get_pwd_(system, user)
