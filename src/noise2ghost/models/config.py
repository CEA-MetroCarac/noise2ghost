"""Models and losses handling tools."""

from autoden.models.config import NetworkParams
from torch.cuda import is_available as is_cuda_available

from noise2ghost.models.inr import SIREN


class NetworkParamsINR(NetworkParams):
    """Store INR parameters."""

    n_embeddings: int
    n_layers: int

    def __init__(
        self,
        n_channels_in: int = 2,
        n_channels_out: int = 1,
        n_layers: int = 3,
        n_features: int = 256,
        n_embeddings: int = 256,
    ) -> None:
        """Initialize the DnCNN network parameters definition.

        Parameters
        ----------
        n_channels_in : int, optional
            Number of input channels. Default is 1.
        n_channels_out : int, optional
            Number of output channels. Default is 1.
        n_layers : int, optional
            Number of layers. Default is 8.
        n_features : int, optional
            Number of features. Default is 128.
        n_embeddings : int, optional
            Number of embeddings. Default is 128.
        """
        super().__init__(n_features=n_features, n_channels_in=n_channels_in, n_channels_out=n_channels_out)
        self.n_layers = n_layers
        self.n_embeddings = n_embeddings

    def get_model(self, device: str = "cuda" if is_cuda_available() else "cpu") -> SIREN:
        """Get a INR model with the selected parameters.

        Parameters
        ----------
        device : str, optional
            The device that the the model should run on, by default "cuda" if cuda is available, otherwise "cpu".

        Returns
        -------
        Module
            The INR model.
        """
        return SIREN(
            n_channels_in=self.n_channels_in,
            n_embeddings=self.n_embeddings,
            n_channels_out=self.n_channels_out,
            n_layers=self.n_layers,
            n_features=self.n_features,
            device=device,
        )
