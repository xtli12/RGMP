from .protocol1_packet_handler import *
from .protocol2_packet_handler import *


def PacketHandler(protocol_version):
    # FIXME: float or int-to-float comparison can generate weird behaviour
    if protocol_version == 1.0:
        return Protocol1PacketHandler()
    elif protocol_version == 2.0:
        return Protocol2PacketHandler()
    else:
        return Protocol2PacketHandler()
