# -*- coding: utf-8 -*-
"""用 Playwright 截取 QIS Terminal 各页面，供视觉检查。"""
import asyncio
import sys

from playwright.async_api import async_playwright

BASE = "http://127.0.0.1:8600"
OUT = "/tmp/qis_shots"


async def main() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1680, "height": 1050})
        errors = []
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: errors.append(str(e)))

        # 1) 总览（策略卡片要跑 3 个回测，首次较慢）
        await page.goto(BASE, wait_until="networkidle")
        await page.wait_for_selector("#stat-row .kpi", timeout=60000)
        await page.wait_for_selector("#strategy-cards .strat-card", timeout=180000)
        await page.wait_for_timeout(2500)
        await page.screenshot(path=f"{OUT}/01_overview.png")

        # 2) 策略回测页（默认 trend 已缓存，直接渲染）
        await page.click('[data-page="strategy"]')
        await page.wait_for_selector("#metric-strip .metric", timeout=120000)
        await page.wait_for_timeout(2500)
        await page.screenshot(path=f"{OUT}/02_strategy.png", full_page=True)

        # 3) 标的池
        await page.click('[data-page="universe"]')
        await page.wait_for_selector("#u-tbody tr", timeout=30000)
        await page.wait_for_timeout(4000)  # 等 sparkline 按需加载
        await page.screenshot(path=f"{OUT}/03_universe.png")

        # 4) 抽屉
        await page.click('#u-tbody tr[data-name="WTI"]')
        await page.wait_for_timeout(2000)
        await page.screenshot(path=f"{OUT}/04_drawer.png")
        await page.click("#drawer-close")

        # 5) 数据状态
        await page.click('[data-page="data"]')
        await page.wait_for_selector("#d-tbody tr", timeout=30000)
        await page.screenshot(path=f"{OUT}/05_data.png")

        if errors:
            print("CONSOLE ERRORS:")
            for e in errors[:10]:
                print("  ", e[:200])
        else:
            print("no console errors")
        await browser.close()


asyncio.run(main())
