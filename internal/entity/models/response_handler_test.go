//
//  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

package models

import (
	"bytes"
	"io"
	"strings"
	"testing"
)

// TestHandleStreamingResponseBareRootFinishReason verifies the shared handler
// terminates the stream when an event carries only a root-level finish_reason
// with no choices and no trailing [DONE]. Before the fix this event was
// skipped by the choices guard and the handler returned "stream ended before
// [DONE] or finish_reason".
func TestHandleStreamingResponseBareRootFinishReason(t *testing.T) {
	sse := "data: {\"finish_reason\":\"stop\"}\n"

	var contentChunks, reasonChunks []string
	err := HandleStreamingResponse(
		io.NopCloser(bytes.NewBufferString(sse)),
		nil,
		nil,
		OpenAIParserConfig,
		func(content *string, reason *string) error {
			if content != nil && *content != "" {
				contentChunks = append(contentChunks, *content)
			}
			if reason != nil && *reason != "" {
				reasonChunks = append(reasonChunks, *reason)
			}
			return nil
		},
	)
	if err != nil {
		t.Fatalf("HandleStreamingResponse: %v", err)
	}

	// The terminal [DONE] marker is the only content the handler emits for a
	// finish_reason-only event: no tokens arrived and no reasoning was sent.
	if got := strings.Join(contentChunks, ""); got != "[DONE]" {
		t.Errorf("content=%q, want only the [DONE] marker", got)
	}
	if len(reasonChunks) != 0 {
		t.Errorf("reasoning=%q, want none for a finish_reason-only event", strings.Join(reasonChunks, ""))
	}
}

// TestHandleStreamingResponseTruncatedStream verifies a stream that ends
// without [DONE], a root-level finish_reason, or a per-choice finish_reason
// is rejected as truncated. This is the counterpart of the bare
// finish_reason case: only a genuine truncation must fail.
func TestHandleStreamingResponseTruncatedStream(t *testing.T) {
	sse := "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n"

	err := HandleStreamingResponse(
		io.NopCloser(bytes.NewBufferString(sse)),
		nil,
		nil,
		OpenAIParserConfig,
		func(*string, *string) error { return nil },
	)
	if err == nil || !strings.Contains(err.Error(), "stream ended before [DONE] or finish_reason") {
		t.Fatalf("expected truncation error, got %v", err)
	}
}

// TestHandleStreamingResponseTerminalChoiceWithoutDelta verifies the terminal
// frame is recognised even when the choice carries nothing but finish_reason.
// MiniMax closes a stream with finish_reason and no [DONE], and that last frame
// is not required to repeat an empty delta; reading the delta first turned every
// such completed stream into a truncation error.
func TestHandleStreamingResponseTerminalChoiceWithoutDelta(t *testing.T) {
	cases := map[string]string{
		"no delta key":     "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\ndata: {\"choices\":[{\"finish_reason\":\"stop\"}]}\n",
		"null delta":       "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\ndata: {\"choices\":[{\"delta\":null,\"finish_reason\":\"stop\"}]}\n",
		"tool_calls":       "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\ndata: {\"choices\":[{\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n",
		"second choice":    "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}},{\"finish_reason\":\"stop\"}]}\n",
		"usage only first": "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"total_tokens\":5}}\n",
	}
	for name, sse := range cases {
		t.Run(name, func(t *testing.T) {
			var content []string
			err := HandleStreamingResponse(
				io.NopCloser(bytes.NewBufferString(sse)),
				nil,
				nil,
				OpenAIParserConfig,
				func(answer *string, _ *string) error {
					if answer != nil && *answer != "" {
						content = append(content, *answer)
					}
					return nil
				},
			)
			if name == "usage only first" {
				// A trailing usage chunk is not a terminal mark: the answer is
				// still incomplete without [DONE] or finish_reason.
				if err == nil {
					t.Fatal("a usage-only trailing chunk was accepted as a terminal frame")
				}
				return
			}
			if err != nil {
				t.Fatalf("HandleStreamingResponse: %v", err)
			}
			if got := strings.Join(content, ""); got != "hi[DONE]" {
				t.Errorf("content=%q, want the streamed answer then [DONE]", got)
			}
		})
	}
}
