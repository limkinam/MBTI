from flask_restful import Resource
from ldap3 import Server, Connection
 
class LoginAPI(Resource):
    def login(self, ldap_info):
        return True
