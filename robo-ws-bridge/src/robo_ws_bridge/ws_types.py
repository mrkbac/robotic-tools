"""Type definitions for WebSocket protocol messages."""

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Literal, TypedDict

from typing_extensions import NotRequired


class ConnectionStatus(Enum):
    """Connection status states."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"


class JsonOpCodes(Enum):
    """JSON WebSocket operation codes."""

    # Server messages
    SERVER_INFO = "serverInfo"
    STATUS = "status"
    REMOVE_STATUS = "removeStatus"
    ADVERTISE = "advertise"
    UNADVERTISE = "unadvertise"
    PARAMETER_VALUES = "parameterValues"
    ADVERTISE_SERVICES = "advertiseServices"
    UNADVERTISE_SERVICES = "unadvertiseServices"
    CONNECTION_GRAPH_UPDATE = "connectionGraphUpdate"
    SERVICE_CALL_FAILURE = "serviceCallFailure"

    # Client messages
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    GET_PARAMETERS = "getParameters"
    SET_PARAMETERS = "setParameters"
    SUBSCRIBE_PARAMETER_UPDATES = "subscribeParameterUpdates"
    UNSUBSCRIBE_PARAMETER_UPDATES = "unsubscribeParameterUpdates"
    SUBSCRIBE_CONNECTION_GRAPH = "subscribeConnectionGraph"
    UNSUBSCRIBE_CONNECTION_GRAPH = "unsubscribeConnectionGraph"
    FETCH_ASSET = "fetchAsset"


class BinaryOpCodes(IntEnum):
    """Binary WebSocket operation codes."""

    MESSAGE_DATA = 0x01
    TIME = 0x02
    SERVICE_CALL_RESPONSE = 0x03
    FETCH_ASSET_RESPONSE = 0x04
    PLAYBACK_STATE = 0x05
    CLIENT_MESSAGE_DATA = 0x01  # Same as MESSAGE_DATA
    SERVICE_CALL_REQUEST = 0x02  # Same as TIME
    PLAYBACK_CONTROL_REQUEST = 0x03  # Same as SERVICE_CALL_RESPONSE


class StatusLevel(IntEnum):
    """Status message levels."""

    INFO = 0
    WARNING = 1
    ERROR = 2


class ServerCapabilities(Enum):
    """Server capability strings."""

    CLIENT_PUBLISH = "clientPublish"
    PARAMETERS = "parameters"
    PARAMETERS_SUBSCRIBE = "parametersSubscribe"
    TIME = "time"
    SERVICES = "services"
    CONNECTION_GRAPH = "connectionGraph"
    ASSETS = "assets"
    PLAYBACK_CONTROL = "playbackControl"


class PlaybackCommand(IntEnum):
    """Playback commands sent by a Foxglove client."""

    PLAY = 0
    PAUSE = 1


class PlaybackStatus(IntEnum):
    """Current status of server-side playback."""

    PLAYING = 0
    PAUSED = 1
    BUFFERING = 2
    ENDED = 3


@dataclass(frozen=True, slots=True)
class PlaybackControlRequest:
    """A decoded client request to control server-side playback."""

    playback_command: PlaybackCommand
    playback_speed: float
    seek_time: int | None
    request_id: str


@dataclass(frozen=True, slots=True)
class PlaybackState:
    """Playback state broadcast by the server."""

    status: PlaybackStatus
    current_time: int
    playback_speed: float
    did_seek: bool
    request_id: str | None = None


# Type definitions for JSON messages

ParameterValue = float | bool | str | list["ParameterValue"] | dict[str, "ParameterValue"] | None


class ServerInfoMessage(TypedDict):
    """Server Info message."""

    op: Literal["serverInfo"]
    name: str
    capabilities: list[str]
    supportedEncodings: NotRequired[list[str]]
    metadata: NotRequired[dict[str, str]]
    sessionId: NotRequired[str]
    dataStartTime: NotRequired["SerializedTimestamp"]
    dataEndTime: NotRequired["SerializedTimestamp"]


class SerializedTimestamp(TypedDict):
    """JSON timestamp used for playback time ranges."""

    sec: int
    nsec: int


class StatusMessage(TypedDict):
    """Status message."""

    op: Literal["status"]
    level: int  # 0=info, 1=warning, 2=error
    message: str
    id: NotRequired[str]


class RemoveStatusMessage(TypedDict):
    """Remove Status message."""

    op: Literal["removeStatus"]
    statusIds: list[str]


class ChannelInfo(TypedDict):
    """Channel information for advertise messages."""

    id: int
    topic: str
    encoding: str
    schemaName: str
    schema: str
    schemaEncoding: NotRequired[str]


class AdvertiseMessage(TypedDict):
    """Advertise message."""

    op: Literal["advertise"]
    channels: list[ChannelInfo]


class UnadvertiseMessage(TypedDict):
    """Unadvertise message."""

    op: Literal["unadvertise"]
    channelIds: list[int]


class Parameter(TypedDict):
    """Parameter definition."""

    name: str
    value: ParameterValue
    type: NotRequired[Literal["byte_array", "float64", "float64_array"]]


class ParameterValuesMessage(TypedDict):
    """Parameter Values message."""

    op: Literal["parameterValues"]
    parameters: list[Parameter]
    id: NotRequired[str]


class ServiceSchema(TypedDict):
    """Service schema definition."""

    encoding: str
    schemaName: str
    schemaEncoding: str
    schema: str


class ServiceInfo(TypedDict):
    """Service information for advertise services."""

    id: int
    name: str
    type: str
    request: NotRequired[ServiceSchema]
    response: NotRequired[ServiceSchema]
    requestSchema: NotRequired[str]  # Deprecated
    responseSchema: NotRequired[str]  # Deprecated


class AdvertiseServicesMessage(TypedDict):
    """Advertise Services message."""

    op: Literal["advertiseServices"]
    services: list[ServiceInfo]


class UnadvertiseServicesMessage(TypedDict):
    """Unadvertise Services message."""

    op: Literal["unadvertiseServices"]
    serviceIds: list[int]


class PublishedTopic(TypedDict):
    """Published topic information."""

    name: str
    publisherIds: list[str]


class SubscribedTopic(TypedDict):
    """Subscribed topic information."""

    name: str
    subscriberIds: list[str]


class AdvertisedService(TypedDict):
    """Advertised service information."""

    name: str
    providerIds: list[str]


class ConnectionGraphUpdateMessage(TypedDict):
    """Connection Graph Update message."""

    op: Literal["connectionGraphUpdate"]
    publishedTopics: NotRequired[list[PublishedTopic]]
    subscribedTopics: NotRequired[list[SubscribedTopic]]
    advertisedServices: NotRequired[list[AdvertisedService]]
    removedTopics: NotRequired[list[str]]
    removedServices: NotRequired[list[str]]


class ServiceCallFailureMessage(TypedDict):
    """Service Call Failure message."""

    op: Literal["serviceCallFailure"]
    serviceId: int
    callId: int
    message: str


class SubscriptionInfo(TypedDict):
    """Subscription information."""

    id: int
    channelId: int


class SubscribeMessage(TypedDict):
    """Subscribe message."""

    op: Literal["subscribe"]
    subscriptions: list[SubscriptionInfo]


class UnsubscribeMessage(TypedDict):
    """Unsubscribe message."""

    op: Literal["unsubscribe"]
    subscriptionIds: list[int]


class GetParametersMessage(TypedDict):
    """Get Parameters message."""

    op: Literal["getParameters"]
    parameterNames: list[str]
    id: NotRequired[str]


class SetParametersMessage(TypedDict):
    """Set Parameters message."""

    op: Literal["setParameters"]
    parameters: list[Parameter]
    id: NotRequired[str]


class SubscribeParameterUpdatesMessage(TypedDict):
    """Subscribe Parameter Updates message."""

    op: Literal["subscribeParameterUpdates"]
    parameterNames: list[str]


class UnsubscribeParameterUpdatesMessage(TypedDict):
    """Unsubscribe Parameter Updates message."""

    op: Literal["unsubscribeParameterUpdates"]
    parameterNames: list[str]


class SubscribeConnectionGraphMessage(TypedDict):
    """Subscribe Connection Graph message."""

    op: Literal["subscribeConnectionGraph"]


class UnsubscribeConnectionGraphMessage(TypedDict):
    """Unsubscribe Connection Graph message."""

    op: Literal["unsubscribeConnectionGraph"]


class FetchAssetMessage(TypedDict):
    """Fetch Asset message."""

    op: Literal["fetchAsset"]
    uri: str
    requestId: int


# Union type for all possible JSON messages
JsonMessage = (
    ServerInfoMessage
    | StatusMessage
    | RemoveStatusMessage
    | AdvertiseMessage
    | UnadvertiseMessage
    | ParameterValuesMessage
    | AdvertiseServicesMessage
    | UnadvertiseServicesMessage
    | ConnectionGraphUpdateMessage
    | ServiceCallFailureMessage
    | SubscribeMessage
    | UnsubscribeMessage
    | GetParametersMessage
    | SetParametersMessage
    | SubscribeParameterUpdatesMessage
    | UnsubscribeParameterUpdatesMessage
    | SubscribeConnectionGraphMessage
    | UnsubscribeConnectionGraphMessage
    | FetchAssetMessage
)
