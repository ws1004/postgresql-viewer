"""CRC-32C (Castagnoli). PostgreSQL 의 COMP_CRC32C / FIN_CRC32C 와 같은 결과.

src/include/port/pg_crc32c.h: 초기값 0xFFFFFFFF, 반사 다항식 0x82F63B78, 최종 XOR 0xFFFFFFFF.
"""

_POLY = 0x82F63B78


def _make_table() -> list[int]:
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = (c >> 1) ^ _POLY if c & 1 else c >> 1
        table.append(c)
    return table


_TABLE = _make_table()


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    t = _TABLE
    for b in data:
        crc = t[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF
