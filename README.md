# SWV2 Story Search

## Stateless chat API

`POST /v1/story-chat` is a client-managed conversation endpoint. Send the full
recent transcript (up to 20 messages) and include the shared testing token:

```http
Authorization: Bearer <CHAT_API_TOKEN>
Content-Type: application/json
```

```json
{
  "messages": [
    { "role": "assistant", "content": "Hello! What kind of story would you like?" },
    { "role": "user", "content": "Find a short animal story" }
  ]
}
```

The response is always `{ "response": "..." }`. Append that text locally as
an `assistant` message before sending the next user message. Configure one
fixed, cryptographically random `CHAT_API_TOKEN` in your deployment secrets;
the API does not retain chat sessions or transcripts.
