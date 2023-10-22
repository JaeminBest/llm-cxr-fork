from taming.models.vqgan import VQModel
import torch
import pytorch_lightning as pl


class VQModelDecoderWrapper(VQModel):
    def __init__(
        self,
        ddconfig=None,
        lossconfig=None,
        n_embed=None,
        embed_dim=None,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
    ):
        super().__init__(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            n_embed=n_embed,
            embed_dim=embed_dim,
            ckpt_path=ckpt_path,
            ignore_keys=ignore_keys,
            image_key=image_key,
            colorize_nlabels=colorize_nlabels,
            monitor=monitor,
        )

    def forward(self, indices: torch.Tensor | list[int]) -> torch.tensor:
        if isinstance(indices, list):
            indices = torch.tensor(indices)
        indices = indices.to(self.device)
        quant = self.quantize.get_codebook_entry(indices, shape=(1, 16, 16, -1))
        img = self.decode(quant.to(self.device))
        img = img.squeeze().permute(1, 2, 0).detach()
        img = torch.clip(
            img,
            min=torch.tensor(-1.0).to(self.device),
            max=torch.tensor(1.0).to(self.device),
            out=img,
        )
        img = (img + 1.0) / 2.0
        return img


class VQModelEncoderWrapper(VQModel):
    def __init__(
        self,
        ddconfig=None,
        lossconfig=None,
        n_embed=None,
        embed_dim=None,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
    ):
        super().__init__(
            ddconfig=ddconfig,
            lossconfig=lossconfig,
            n_embed=n_embed,
            embed_dim=embed_dim,
            ckpt_path=ckpt_path,
            ignore_keys=ignore_keys,
            image_key=image_key,
            colorize_nlabels=colorize_nlabels,
            monitor=monitor,
        )

    def forward(self, indices: torch.Tensor | list[int]) -> torch.tensor:
        if isinstance(indices, list):
            indices = torch.tensor(indices)
        indices = indices.to(self.device)
        img = 2.0 * indices - 1.0
        _, _, [_, _, indices] = self.encode(img.unsqueeze(0))
        indices = indices.reshape(1, -1).squeeze()
        return indices
