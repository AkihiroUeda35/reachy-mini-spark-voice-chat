from __future__ import annotations

import asyncio
import json
import os
import unicodedata
from collections.abc import Mapping
from typing import Any

import httpx
from langchain_core.tools import tool


JMA_AREA_URL = "https://www.jma.go.jp/bosai/common/const/area.json"
JMA_FORECAST_URL = "https://www.jma.go.jp/bosai/forecast/data/forecast/{office_code}.json"
JMA_OVERVIEW_URL = "https://www.jma.go.jp/bosai/forecast/data/overview_forecast/{office_code}.json"
JMA_TIMEOUT = httpx.Timeout(20.0)
DEFAULT_CITY = os.environ.get("JMA_DEFAULT_CITY", "長岡京市")

_AREA_CACHE: dict[str, Any] | None = None
_AREA_CACHE_LOCK = asyncio.Lock()


def _normalize_name(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().replace(" ", "").replace("　", "")


def _requested_city_variants(city: str) -> set[str]:
    normalized = _normalize_name(city or DEFAULT_CITY)
    variants = {normalized}
    if normalized and normalized[-1] not in {"市", "町", "村", "区"}:
        variants.update({f"{normalized}市", f"{normalized}町", f"{normalized}村", f"{normalized}区"})
    return variants


def _build_area_index(area_data: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for group in ("centers", "offices", "class10s", "class15s", "class20s"):
        for code, node in area_data.get(group, {}).items():
            indexed = dict(node)
            indexed["code"] = code
            indexed["group"] = group
            index[code] = indexed
    return index


def _find_municipality(area_data: Mapping[str, Any], city: str) -> tuple[str, dict[str, Any]]:
    requested = _requested_city_variants(city)
    class20s = area_data.get("class20s", {})

    for code, node in class20s.items():
        if _normalize_name(node.get("name", "")) in requested:
            return code, dict(node)

    partial_matches: list[tuple[str, dict[str, Any]]] = []
    for code, node in class20s.items():
        normalized_name = _normalize_name(node.get("name", ""))
        if any(candidate and candidate in normalized_name for candidate in requested):
            partial_matches.append((code, dict(node)))

    if len(partial_matches) == 1:
        return partial_matches[0]

    raise ValueError(f"JMA area.json could not uniquely resolve municipality '{city}'.")


def _ancestor_codes(area_index: Mapping[str, Mapping[str, Any]], code: str) -> list[str]:
    ancestors: list[str] = []
    current = code
    seen: set[str] = set()

    while current and current not in seen:
        seen.add(current)
        node = area_index.get(current)
        if node is None:
            break
        ancestors.append(current)
        current = str(node.get("parent") or "")

    return ancestors


def _find_office(area_index: Mapping[str, Mapping[str, Any]], code: str) -> tuple[str, Mapping[str, Any]]:
    for ancestor_code in _ancestor_codes(area_index, code):
        node = area_index.get(ancestor_code)
        if node and node.get("group") == "offices":
            return ancestor_code, node
    raise ValueError("JMA office code could not be resolved from area hierarchy.")


def _available_forecast_areas(forecast_payload: list[dict[str, Any]]) -> dict[str, str]:
    areas: dict[str, str] = {}
    for block in forecast_payload:
        for time_series in block.get("timeSeries", []):
            for area in time_series.get("areas", []):
                area_info = area.get("area", {})
                code = str(area_info.get("code") or "")
                if code:
                    areas[code] = str(area_info.get("name") or code)
    return areas


def _resolve_forecast_area(
    area_index: Mapping[str, Mapping[str, Any]],
    municipality_code: str,
    forecast_payload: list[dict[str, Any]],
) -> tuple[str, str]:
    available_areas = _available_forecast_areas(forecast_payload)
    for ancestor_code in _ancestor_codes(area_index, municipality_code):
        if ancestor_code in available_areas:
            return ancestor_code, available_areas[ancestor_code]
    raise ValueError("JMA forecast area could not be resolved for the requested municipality.")


def _first_series_for_area(
    forecast_payload: list[dict[str, Any]],
    area_code: str,
    field_name: str,
) -> tuple[list[str], dict[str, Any]] | None:
    for block in forecast_payload:
        for time_series in block.get("timeSeries", []):
            for area in time_series.get("areas", []):
                area_info = area.get("area", {})
                if str(area_info.get("code") or "") == area_code and field_name in area:
                    return list(time_series.get("timeDefines", [])), area
    return None


def _date_only(timestamp: str) -> str:
    return timestamp.split("T", 1)[0]


def _extract_daily_weathers(forecast_payload: list[dict[str, Any]], area_code: str) -> dict[str, dict[str, Any]]:
    series = _first_series_for_area(forecast_payload, area_code, "weathers")
    if series is None:
        return {}

    time_defines, area = series
    daily: dict[str, dict[str, Any]] = {}
    for timestamp, weather in zip(time_defines, area.get("weathers", [])):
        if not weather:
            continue
        day = _date_only(timestamp)
        daily.setdefault(day, {})["weather"] = str(weather)
    return daily


def _merge_precipitation_probabilities(
    forecast_payload: list[dict[str, Any]],
    area_code: str,
    daily: dict[str, dict[str, Any]],
) -> None:
    series = _first_series_for_area(forecast_payload, area_code, "pops")
    if series is None:
        return

    time_defines, area = series
    grouped: dict[str, list[int]] = {}
    for timestamp, value in zip(time_defines, area.get("pops", [])):
        if value in (None, ""):
            continue
        day = _date_only(timestamp)
        grouped.setdefault(day, []).append(int(value))

    for day, values in grouped.items():
        if day in daily and values:
            daily[day]["precipitation_probability_max"] = max(values)


def _merge_representative_temperatures(
    forecast_payload: list[dict[str, Any]],
    daily: dict[str, dict[str, Any]],
) -> str | None:
    if len(forecast_payload) < 2:
        return None

    weekly_block = forecast_payload[1]
    representative_area: dict[str, Any] | None = None
    time_defines: list[str] = []

    for time_series in weekly_block.get("timeSeries", []):
        if not time_series.get("areas"):
            continue
        first_area = time_series["areas"][0]
        if "tempsMin" in first_area and "tempsMax" in first_area:
            representative_area = first_area
            time_defines = list(time_series.get("timeDefines", []))
            break

    if representative_area is None:
        return None

    point_name = str(representative_area.get("area", {}).get("name") or "")
    mins = representative_area.get("tempsMin", [])
    maxes = representative_area.get("tempsMax", [])

    for timestamp, min_temp, max_temp in zip(time_defines, mins, maxes):
        day = _date_only(timestamp)
        if day not in daily:
            continue
        if min_temp not in (None, ""):
            daily[day]["temp_min_c"] = int(min_temp)
        if max_temp not in (None, ""):
            daily[day]["temp_max_c"] = int(max_temp)

    return point_name or None


async def _load_area_data(client: httpx.AsyncClient) -> dict[str, Any]:
    global _AREA_CACHE
    if _AREA_CACHE is not None:
        return _AREA_CACHE

    async with _AREA_CACHE_LOCK:
        if _AREA_CACHE is None:
            response = await client.get(JMA_AREA_URL)
            response.raise_for_status()
            _AREA_CACHE = response.json()
    assert _AREA_CACHE is not None
    return _AREA_CACHE


async def get_jma_weather(city: str = DEFAULT_CITY) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=JMA_TIMEOUT) as client:
        area_data = await _load_area_data(client)
        area_index = _build_area_index(area_data)
        municipality_code, municipality = _find_municipality(area_data, city or DEFAULT_CITY)
        office_code, office = _find_office(area_index, municipality_code)

        forecast_response, overview_response = await asyncio.gather(
            client.get(JMA_FORECAST_URL.format(office_code=office_code)),
            client.get(JMA_OVERVIEW_URL.format(office_code=office_code)),
        )
        forecast_response.raise_for_status()
        overview_response.raise_for_status()

    forecast_payload = forecast_response.json()
    overview_payload = overview_response.json()

    forecast_area_code, forecast_area_name = _resolve_forecast_area(area_index, municipality_code, forecast_payload)
    daily = _extract_daily_weathers(forecast_payload, forecast_area_code)
    _merge_precipitation_probabilities(forecast_payload, forecast_area_code, daily)
    temperature_point = _merge_representative_temperatures(forecast_payload, daily)

    ancestor_codes = _ancestor_codes(area_index, municipality_code)
    subregion_code = ancestor_codes[1] if len(ancestor_codes) > 1 else municipality_code
    subregion_name = str(area_index.get(subregion_code, {}).get("name") or municipality.get("name") or city)

    forecast_days = []
    for day in sorted(daily.keys())[:2]:
        entry = {"date": day, **daily[day]}
        if temperature_point and any(key in entry for key in ("temp_min_c", "temp_max_c")):
            entry["temperature_point"] = temperature_point
        forecast_days.append(entry)

    return {
        "location": {
            "requested": city or DEFAULT_CITY,
            "municipality": str(municipality.get("name") or city or DEFAULT_CITY),
            "municipality_code": municipality_code,
            "subregion": subregion_name,
            "forecast_area": forecast_area_name,
            "forecast_area_code": forecast_area_code,
            "office": str(office.get("name") or office_code),
            "office_code": office_code,
            "forecast_office_name": str(office.get("officeName") or office.get("name") or office_code),
        },
        "report_datetime": forecast_payload[0].get("reportDatetime"),
        "overview_report_datetime": overview_payload.get("reportDatetime"),
        "overview_text": str(overview_payload.get("text") or "").strip(),
        "forecast": forecast_days,
        "notes": [
            f"The municipality forecast is mapped to JMA area '{forecast_area_name}'.",
            "Temperature values use the representative JMA point when available.",
        ],
    }


def format_jma_weather_report(report: Mapping[str, Any]) -> str:
    location = report.get("location", {})
    lines = [
        f"対象: {location.get('municipality', DEFAULT_CITY)}",
        f"予報地域: {location.get('subregion', '')} / {location.get('forecast_area', '')}",
    ]
    report_datetime = report.get("report_datetime")
    if report_datetime:
        lines.append(f"発表時刻: {report_datetime}")

    for entry in report.get("forecast", []):
        parts = [f"{entry.get('date')}: {entry.get('weather', '')}"]
        if entry.get("precipitation_probability_max") is not None:
            parts.append(f"降水確率最大 {entry['precipitation_probability_max']}%")
        if entry.get("temp_min_c") is not None:
            parts.append(f"最低 {entry['temp_min_c']}度")
        if entry.get("temp_max_c") is not None:
            parts.append(f"最高 {entry['temp_max_c']}度")
        if entry.get("temperature_point"):
            parts.append(f"代表地点 {entry['temperature_point']}")
        lines.append(" / ".join(parts))

    overview_text = str(report.get("overview_text") or "").strip()
    if overview_text:
        lines.append(f"概況: {overview_text}")
    return "\n".join(lines)


@tool
async def get_jma_weather_tool(city: str = DEFAULT_CITY) -> str:
    """Get the latest JMA weather forecast for a Japanese municipality."""
    city = str(city or DEFAULT_CITY).strip() or DEFAULT_CITY
    try:
        result = await get_jma_weather(city)
    except Exception as exc:
        return json.dumps({"error": f"Failed to fetch JMA weather: {exc}", "requested": city}, ensure_ascii=False)
    return format_jma_weather_report(result)


get_jma_weather_tool.description = (
    f"Get the latest JMA weather forecast for a Japanese municipality. "
    f"Omit city to use {DEFAULT_CITY}."
)

LANGGRAPH_TOOLS = [get_jma_weather_tool]