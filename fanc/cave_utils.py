from caveclient import CAVEclient
from .soma_table_utils import SomaTableOrganizer

CAVE_DATASETS = {'production': 'fanc_production_mar2021',
                 'sandbox': 'fanc_sandbox'}

class CAVEorganizer(object):
    def __init__(
        self,
        client,
        datastack_name=None,
        server_address=None,
        auth_token_file=None,
        auth_token_key=None,
        auth_token=None,
        global_only=False,
        pool_maxsize=None,
        pool_block=None,
        desired_resolution=None,
        info_cache=None
    ):
        if client is None:
            self.client = CAVEclient(datastack_name=datastack_name,
                                     server_address=server_address,
                                     auth_token_file=auth_token_file,
                                     auth_token_key=auth_token_key,
                                     auth_token=auth_token,
                                     global_only=global_only,
                                     pool_maxsize=pool_maxsize,
                                     pool_block=pool_block,
                                     desired_resolution=desired_resolution,
                                     info_cache=info_cache)
        else:
            self._client = client
        self._reset_services()

    def _reset_services(self):
        self._update_soma = None

    @property
    def client(self):
        return self._client

    def get_tables(self, datastack_name=None, version=None):
        return self._client.materialize.get_tables(datastack_name=datastack_name, 
                                                   version=version)

    def get_info(self, datastack_name=None):
        return self._client.info.get_datastack_info(datastack_name=datastack_name)

    @property
    def update_soma(self):
        if self._update_soma is None:
            self._update_soma = SomaTableOrganizer(
                client=self._client
            )
        return self._update_soma