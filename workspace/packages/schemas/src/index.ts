import { z } from 'zod';

// =============================================================================
// Session & Call Schemas
// =============================================================================

export const SessionIdSchema = z.string().uuid();
export const CallIdSchema = z.string().uuid();
export const TurnIdSchema = z.string().uuid();

export const LanguageCodeSchema = z.enum(['en-IN', 'hi-IN', 'mr-IN']);
export type LanguageCode = z.infer<typeof LanguageCodeSchema>;

export const CallStatusSchema = z.enum(['active', 'completed', 'failed', 'transferred']);
export type CallStatus = z.infer<typeof CallStatusSchema>;

export const TurnRoleSchema = z.enum(['user', 'assistant', 'system']);
export type TurnRole = z.infer<typeof TurnRoleSchema>;

export const DecisionTypeSchema = z.enum(['answer', 'clarify', 'abstain', 'handoff']);
export type DecisionType = z.infer<typeof DecisionTypeSchema>;

// =============================================================================
// WebSocket Events (Browser & Exotel Transport)
// =============================================================================

// Client -> Server Events
export const ClientConnectEventSchema = z.object({
  type: z.literal('connect'),
  sessionToken: z.string().optional(),
  metadata: z.record(z.unknown()).optional(),
});

export const ClientAudioEventSchema = z.object({
  type: z.literal('audio'),
  data: z.string(), // Base64 PCM16 mono 16kHz
  isFinal: z.boolean().optional(),
});

export const ClientTextEventSchema = z.object({
  type: z.literal('text'),
  text: z.string(),
  language: LanguageCodeSchema.optional(),
});

export const ClientInterruptEventSchema = z.object({
  type: z.literal('interrupt'),
});

export const ClientEndEventSchema = z.object({
  type: z.literal('end'),
});

export const ClientEventSchema = z.discriminatedUnion('type', [
  ClientConnectEventSchema,
  ClientAudioEventSchema,
  ClientTextEventSchema,
  ClientInterruptEventSchema,
  ClientEndEventSchema,
]);

export type ClientEvent = z.infer<typeof ClientEventSchema>;

// Server -> Client Events
export const ServerReadyEventSchema = z.object({
  type: z.literal('ready'),
  sessionId: SessionIdSchema,
  supportedLanguages: z.array(LanguageCodeSchema),
});

export const ServerTranscriptEventSchema = z.object({
  type: z.literal('transcript'),
  text: z.string(),
  isFinal: z.boolean(),
  language: LanguageCodeSchema.optional(),
  confidence: z.number().optional(),
});

export const ServerLanguageDetectedEventSchema = z.object({
  type: z.literal('language_detected'),
  language: LanguageCodeSchema,
  confidence: z.number(),
});

export const ServerThinkingEventSchema = z.object({
  type: z.literal('thinking'),
});

export const ServerResponseEventSchema = z.object({
  type: z.literal('response'),
  text: z.string(),
  language: LanguageCodeSchema,
  decisionType: DecisionTypeSchema,
  citations: z.array(z.object({
    sourceId: z.string(),
    chunkId: z.string(),
    text: z.string(),
    confidence: z.number(),
  })).optional(),
});

export const ServerAudioEventSchema = z.object({
  type: z.literal('audio'),
  data: z.string(), // Base64 PCM16 mono 16kHz
  generationId: z.string(),
});

export const ServerAudioMarkEventSchema = z.object({
  type: z.literal('audio_mark'),
  markId: z.string(),
  generationId: z.string(),
});

export const ServerHandoffEventSchema = z.object({
  type: z.literal('handoff'),
  reason: z.string(),
  school: z.string().optional(),
  contactNumber: z.string().optional(),
  summary: z.string(),
});

export const ServerErrorEventSchema = z.object({
  type: z.literal('error'),
  code: z.string(),
  message: z.string(),
  recoverable: z.boolean(),
});

export const ServerEndEventSchema = z.object({
  type: z.literal('end'),
  reason: z.string(),
});

export const ServerLatencyEventSchema = z.object({
  type: z.literal('latency'),
  stage: z.string(),
  durationMs: z.number(),
  timestamp: z.string(),
});

export const ServerEventSchema = z.discriminatedUnion('type', [
  ServerReadyEventSchema,
  ServerTranscriptEventSchema,
  ServerLanguageDetectedEventSchema,
  ServerThinkingEventSchema,
  ServerResponseEventSchema,
  ServerAudioEventSchema,
  ServerAudioMarkEventSchema,
  ServerHandoffEventSchema,
  ServerErrorEventSchema,
  ServerEndEventSchema,
  ServerLatencyEventSchema,
]);

export type ServerEvent = z.infer<typeof ServerEventSchema>;

// =============================================================================
// Knowledge Base Schemas
// =============================================================================

export const PublicationStatusSchema = z.enum(['draft', 'staged', 'published', 'archived']);
export type PublicationStatus = z.infer<typeof PublicationStatusSchema>;

export const SourceTypeSchema = z.enum(['webpage', 'pdf', 'manual', 'json_import']);
export type SourceType = z.infer<typeof SourceTypeSchema>;

export const SchoolSchema = z.enum([
  'STME',  // School of Technology, Management & Engineering
  'SPTM',  // School of Pharmacy & Technology Management
  'SC',    // School of Commerce
  'GENERAL'
]);
export type School = z.infer<typeof SchoolSchema>;

export const KnowledgeChunkSchema = z.object({
  id: z.string().uuid(),
  documentId: z.string().uuid(),
  content: z.string(),
  embedding: z.array(z.number()).optional(),
  metadata: z.object({
    school: SchoolSchema.optional(),
    programName: z.string().optional(),
    academicYear: z.string().optional(),
    category: z.string().optional(),
    pageNumber: z.number().optional(),
    heading: z.string().optional(),
    sourceUrl: z.string().url().optional(),
  }),
  status: PublicationStatusSchema,
  createdAt: z.string().datetime(),
  updatedAt: z.string().datetime(),
});

export type KnowledgeChunk = z.infer<typeof KnowledgeChunkSchema>;

// =============================================================================
// Retrieval & Citation Schemas
// =============================================================================

export const RetrievalResultSchema = z.object({
  chunkId: z.string().uuid(),
  content: z.string(),
  score: z.number(),
  metadata: z.record(z.unknown()),
});

export type RetrievalResult = z.infer<typeof RetrievalResultSchema>;

export const CitationSchema = z.object({
  sourceId: z.string(),
  chunkId: z.string(),
  text: z.string(),
  confidence: z.number(),
  metadata: z.record(z.unknown()).optional(),
});

export type Citation = z.infer<typeof CitationSchema>;

// =============================================================================
// Admin API Schemas
// =============================================================================

export const ContactRouteSchema = z.object({
  id: z.string().uuid(),
  school: SchoolSchema,
  primaryPhone: z.string(),
  fallbackPhones: z.array(z.string()),
  email: z.string().email().optional(),
  businessHours: z.object({
    start: z.string(), // HH:MM format
    end: z.string(),   // HH:MM format
    days: z.array(z.number().min(0).max(6)), // 0 = Sunday
  }).optional(),
  isActive: z.boolean(),
});

export type ContactRoute = z.infer<typeof ContactRouteSchema>;

export const IngestionRunSchema = z.object({
  id: z.string().uuid(),
  status: z.enum(['pending', 'processing', 'completed', 'failed']),
  sourceType: SourceTypeSchema,
  startedAt: z.string().datetime(),
  completedAt: z.string().datetime().optional(),
  documentsProcessed: z.number(),
  chunksCreated: z.number(),
  errors: z.array(z.object({
    url: z.string().optional(),
    error: z.string(),
    timestamp: z.string().datetime(),
  })),
});

export type IngestionRun = z.infer<typeof IngestionRunSchema>;

// =============================================================================
// Analytics Schemas
// =============================================================================

export const CallAnalyticsSchema = z.object({
  totalCalls: z.number(),
  completedCalls: z.number(),
  transferredCalls: z.number(),
  failedCalls: z.number(),
  averageDuration: z.number(), // seconds
  languageDistribution: z.record(LanguageCodeSchema, z.number()),
  schoolDistribution: z.record(SchoolSchema, z.number()),
  p50LatencyMs: z.number(),
  p95LatencyMs: z.number(),
  p99LatencyMs: z.number(),
});

export type CallAnalytics = z.infer<typeof CallAnalyticsSchema>;

// =============================================================================
// Error Schemas
// =============================================================================

export const ErrorResponseSchema = z.object({
  error: z.string(),
  code: z.string(),
  message: z.string(),
  details: z.record(z.unknown()).optional(),
  timestamp: z.string().datetime(),
  requestId: z.string().uuid().optional(),
});

export type ErrorResponse = z.infer<typeof ErrorResponseSchema>;
