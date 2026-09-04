import unittest

from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import ValidationError

from src.search.app import _require_chat_token
from src.search.librarian import client_message_history
from src.search.schemas import ChatRequest, ChatResponse


class ChatRequestTests(unittest.TestCase):
    def test_accepts_client_held_transcript(self):
        request = ChatRequest.model_validate({
            "messages": [
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Find a short animal story"},
            ]
        })

        self.assertEqual(request.messages[-1].content, "Find a short animal story")
        self.assertEqual(len(client_message_history(request.messages[:-1])), 1)

    def test_rejects_transcript_that_does_not_end_with_user(self):
        with self.assertRaises(ValidationError):
            ChatRequest.model_validate({"messages": [{"role": "assistant", "content": "Hello"}]})

    def test_rejects_more_than_twenty_messages(self):
        with self.assertRaises(ValidationError):
            ChatRequest.model_validate({"messages": [{"role": "user", "content": "hello"}] * 21})

    def test_response_exposes_only_response_text(self):
        self.assertEqual(ChatResponse(response="A story suggestion").model_dump(), {"response": "A story suggestion"})


class ChatAuthenticationTests(unittest.TestCase):
    def test_accepts_matching_bearer_token(self):
        _require_chat_token(HTTPAuthorizationCredentials(scheme="Bearer", credentials="test-token"), "test-token")

    def test_rejects_missing_or_wrong_token(self):
        for credentials in (None, HTTPAuthorizationCredentials(scheme="Bearer", credentials="wrong")):
            with self.assertRaises(HTTPException) as raised:
                _require_chat_token(credentials, "test-token")
            self.assertEqual(raised.exception.status_code, 401)
            self.assertEqual(raised.exception.headers, {"WWW-Authenticate": "Bearer"})


if __name__ == "__main__":
    unittest.main()
