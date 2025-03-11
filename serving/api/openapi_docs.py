"""
OpenAPI documentation details for the Roman API.

This module contains all the example responses and request bodies
used in the Swagger UI documentation, keeping them separate from
the main API implementation for better readability.
"""

# Process endpoint documentation
PROCESS_RESPONSES = {
    200: {
        "description": "Tasks successfully scheduled for processing",
        "content": {
            "application/json": {
                "examples": {
                    "Single Input": {
                        "summary": "Processing a single input string",
                        "value": {
                            "batch_id": "d8703998-9f23-4217-a4d1-5c23a8920e55",
                            "message": "Scheduled 1 inputs for processing in batch d8703998-9f23-4217-a4d1-5c23a8920e55",
                        },
                    },
                    "Multiple Inputs": {
                        "summary": "Processing multiple input strings",
                        "value": {
                            "batch_id": "e9814aab-3f52-4c8b-b756-89c2df7af112",
                            "message": "Scheduled 2 inputs for processing in batch e9814aab-3f52-4c8b-b756-89c2df7af112",
                        },
                    },
                }
            }
        },
    }
}

PROCESS_REQUEST_EXAMPLES = {
    "requestBody": {
        "content": {
            "application/json": {
                "examples": {
                    "Single String": {
                        "summary": "Single string example",
                        "value": {"input_string": "boy riding a bike"},
                    },
                    "Array of Strings": {
                        "summary": "Multiple strings example",
                        "value": {
                            "input_string": ["boy riding a bike", "girl riding a bike"]
                        },
                    },
                }
            }
        }
    }
}

# Status endpoint documentation
STATUS_RESPONSES = {
    404: {"description": "Batch ID not found"},
    200: {
        "description": "Successful response with batch status and results",
        "content": {
            "application/json": {
                "examples": {
                    "Completed Batch": {
                        "summary": "All tasks completed successfully",
                        "value": {
                            "ready": True,
                            "tasks": {"SUCCESS": 2},
                            "results": [
                                {
                                    "gcs_link": "https://storage.googleapis.com/bucket/object/abc123",
                                    "input_string": "boy riding a bike",
                                    "task_id": "abc123",
                                },
                                {
                                    "gcs_link": "https://storage.googleapis.com/bucket/object/def456",
                                    "input_string": "girl riding a bike",
                                    "task_id": "def456",
                                },
                            ],
                        },
                    },
                    "Partially Complete Batch": {
                        "summary": "Some tasks still pending",
                        "value": {
                            "ready": False,
                            "tasks": {"SUCCESS": 1, "PENDING": 1},
                            "results": [
                                {
                                    "gcs_link": "https://storage.googleapis.com/bucket/object/abc123",
                                    "input_string": "boy riding a bike",
                                    "task_id": "abc123",
                                },
                                None,
                            ],
                        },
                    },
                }
            }
        },
    },
}

STATUS_REQUEST_EXAMPLES = {
    "requestBody": {
        "content": {
            "application/json": {
                "examples": {
                    "Batch ID Example": {
                        "summary": "Example batch ID",
                        "value": {"batch_id": "d8703998-9f23-4217-a4d1-5c23a8920e55"},
                    }
                }
            }
        }
    }
}

BATCH_REQUEST_EXAMPLES = {
    "requestBody": {
        "content": {
            "application/json": {
                "examples": {
                    "None": {
                    }
                }
            }
        }
    }
}

# Status endpoint documentation
BATCH_RESPONSES = {
    404: {"description": "Batch ID not found"},
    200: {
        "description": "Successful response with batch ids and task results",
        "content": {
            "application/json": {
                "examples": {
                        "summary": "All batch ids",
                        "value": {
                            "batch_id": ["d8703998","9f23","4217","a4d1","5c23a8920e55"],
                            "batches": [{"d8703998": {"status": "PENDING"}},{"5c23a8920e55": {"status": "SUCCESS"}}],
                        }
                }
            }
        }
    }
}
# Health endpoint documentation
HEALTH_RESPONSES = {
    200: {
        "description": "Service is healthy",
        "content": {"application/json": {"example": True}},
    }
}
