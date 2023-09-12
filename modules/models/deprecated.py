

"""
class MultiUnet(nn.Module):
    dim: int = 32
    out_channels: int = 3
    resnet_block_groups: int = 8,
    channels: int = 3,
    dim_mults: Sequence = (1, 2, 4, 8)
    dtype: Any = jnp.bfloat16
    num_unets: int = 4
    self_condition: bool = False

    @nn.compact
    def __call__(self, x, time, x_self_cond=None, *args, **kwargs):
        unet_configs = {
            'dim': self.dim,
            'out_channels': self.out_channels,
            'resnet_block_groups': self.resnet_block_groups,
            'channels': self.channels,
            'dim_mults': self.dim_mults,
            'dtype': self.dtype,
            'self_condition': self.self_condition
        }
        x = Unet(**unet_configs)(x, time, x_self_cond)

        for _ in range(self.num_unets - 1):
            x = Unet(**unet_configs)(x, time, x_self_cond) + x
        return x


class UVit(nn.Module):
    dim: int = 384
    patch: int = 2
    out_channels: int = 3
    depth: int = 12
    # resnet_block_groups: int = 8,
    dtype: Any = jnp.bfloat16
    self_condition: bool = False

    @nn.compact
    def __call__(self, x, time, x_self_cond=None, *args, **kwargs):

        if x_self_cond is not None and self.self_condition:
            x = jnp.concatenate([x, x_self_cond], axis=3)
        elif self.self_condition:
            x = jnp.concatenate([x, jnp.zeros_like(x)], axis=3)
        print(x.shape)

        time_dim = self.dim * 4
        t = nn.Sequential([
            SinusoidalPosEmb(self.dim),
            nn.Dense(time_dim, dtype=self.dtype),
            nn.gelu,
            nn.Dense(time_dim, dtype=self.dtype)
        ])(time)

        x = nn.Conv(self.dim, (self.patch, self.patch), (self.patch, self.patch), padding="SAME", dtype=self.dtype)(x)
        r = x

        h = []

        for _ in range(self.depth // 2):
            x = Transformer(self.dim, self.dtype)(x)
            h.append(x)

        for _ in range(self.depth // 2):
            x = jnp.concatenate([x, h.pop()], axis=-1)
            x = nn.Dense(self.dim, dtype=self.dtype)(x)
            x = Transformer(self.dim, self.dtype)(x)

        x = einops.rearrange(x, 'b h w (c p1 p2)->b (h p1) (w p2) c', p1=self.patch, p2=self.patch)
        x = nn.Conv(self.out_channels, (3, 3), dtype="float32")(x)
        return x


class NAFUnet(nn.Module):
    dim: int = 64
    dim_mults: Sequence = (1, 2, 4, 4)
    num_up_blocks: Any = 2
    num_down_blocks: Any = 2
    num_middle_blocks: Any = 2
    out_channels: int = 3
    resnet_block_groups: int = 8,
    channels: int = 3,
    dtype: Any = jnp.bfloat16
    self_condition: bool = False
    use_encoder: bool = False
    encoder_type: str = '2D'
    res_type: Any = 'default'
    patch_size: int = 1

    @nn.compact
    def __call__(self, x, time, x_self_cond=None, z_rng=None, *args, **kwargs):

        if type(self.num_up_blocks) == int:
            num_up_blocks = (self.num_up_blocks,) * len(self.dim_mults)
        else:
            assert len(self.num_up_blocks) == len(self.dim_mults)
            num_up_blocks = self.num_up_blocks

        if type(self.num_down_blocks) == int:
            num_down_blocks = (self.num_down_blocks,) * len(self.dim_mults)
        else:
            assert len(self.num_down_blocks) == len(self.dim_mults)
            num_down_blocks = self.num_down_blocks

        if self.res_type == 'default':
            res_block = ResBlock
        elif self.res_type == "NAF":
            res_block = NAFBlock
        elif self.res_type == "efficient":
            res_block = EfficientBlock
        else:
            res_block = None

        cond_emb = None
        if self.use_encoder:
            latent = x_self_cond
            assert self.encoder_type in ['1D', '2D', 'Both']
            print(f'latent shape:{latent.shape}')
            if self.encoder_type == '1D':
                cond_emb = latent
                x_self_cond = None
            elif self.encoder_type == '2D':
                cond_emb = None
                x_self_cond = Encoder2DLatent(shape=x.shape)(latent)
            elif self.encoder_type == 'Both':
                cond_emb = nn.Sequential([
                    nn.GroupNorm(num_groups=min(8, latent.shape[-1])),
                    nn.silu,
                    nn.Conv(self.dim * 16, (1, 1)),
                    GlobalAveragePool(),
                    Rearrange('b h w c->b (h w c)'),
                    nn.Dense(self.dim * 16)
                ])(latent)
                x_self_cond = Encoder2DLatent(shape=x.shape)(latent)

        if x_self_cond is not None and self.self_condition:
            x = jnp.concatenate([x, x_self_cond], axis=3)

        print(x.shape)

        time_dim = self.dim * 4
        t = nn.Sequential([
            SinusoidalPosEmb(self.dim),
            nn.Dense(time_dim, dtype=self.dtype),
            nn.gelu,
            nn.Dense(time_dim, dtype=self.dtype)
        ])(time)

        x = einops.rearrange(x, 'b (h p1) (w p2) c->b h w (c p1 p2)', p1=self.patch_size, p2=self.patch_size)

        x = nn.Conv(self.dim, (3, 3), (1, 1), padding="SAME",
                    dtype=self.dtype)(x)
        r = x

        h = [x]

        for i, (dim_mul, num_res_block) in enumerate(zip(self.dim_mults, num_up_blocks)):
            dim = self.dim * dim_mul

            x = nn.Sequential(
                [res_block(dim, dtype=self.dtype) for _ in range(num_res_block)]
            )(x, t, cond_emb)

            h.append(x)

            if i != len(self.dim_mults) - 1:
                x = DownSample(dim, dtype=self.dtype)(x)

            # else:
            #     x = nn.Conv(dim, (3, 3), dtype=self.dtype, padding="SAME")(x)

        for _ in range(self.num_middle_blocks):
            x = res_block(dim, dtype=self.dtype)(x, t, cond_emb)

        reversed_dim_mults = list(reversed(self.dim_mults))

        for i, (dim_mul, num_res_block) in enumerate(zip(reversed_dim_mults, num_down_blocks)):
            dim = self.dim * dim_mul

            x = jnp.concatenate([x, h.pop()], axis=-1)

            x = nn.Sequential(
                [res_block(dim, dtype=self.dtype) for _ in range(num_res_block)]
            )(x, t, cond_emb)

            if i != len(self.dim_mults) - 1:
                x = UpSample(dim, dtype=self.dtype)(x)
            # else:
            #     x = nn.Conv(dim, (3, 3), dtype=self.dtype, padding="SAME")(x)

        # x = jnp.concatenate([x, r], axis=3)
        # x = res_block(dim, dtype=self.dtype)(x, t)
        x = nn.GroupNorm()(x)
        x = nn.silu(x)
        x = nn.Conv(self.out_channels * self.patch_size ** 2, (3, 3), dtype="float32")(x)
        x = einops.rearrange(x, 'b h w (c p1 p2)->b (h p1) (w p2) c', p1=self.patch_size, p2=self.patch_size)

        return x

"""