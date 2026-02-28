#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from secretarius.telegram_adapter import TelegramPollingAdapter


def main() -> int:
    adapter = TelegramPollingAdapter()
    adapter.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

