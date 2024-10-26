# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Fixed

## [0.4.0]

### Added
- Export the flavor `OllamaStream` for Ollama streaming responses.

### Fixed
- Fixes assertion for content-type in `OllamaStream` response (`application/x-ndjson`, not `text/event-stream`).

## [0.3.0]

### Added
- Precompilation statements for OpenAI, Anthropic, and Ollama streaming responses.

## [0.2.0]

### Added
- Added support for Ollama `api/chat` endpoint streaming responses (`flavor = OllamaStream()`).

## [0.1.0]

### Added
- Initial release.